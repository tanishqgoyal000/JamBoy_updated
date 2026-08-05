"""
Production Error-State EKF (ESKF) for GPS-denied navigation, with Mahalanobis gating.

Architecture note
------------------
This follows the same core estimation architecture OpenVINS (and most modern
VIO/VINS stacks) use for their propagation stage:

  - Nominal state is propagated NONLINEARLY by direct IMU mechanization
    (gyro -> attitude quaternion, accel -> velocity/position, in the
    correct rotated frame).
  - A 15-dim ERROR STATE (position, velocity, small-angle attitude error,
    gyro bias, accel bias) is propagated linearly and used only to compute
    the Kalman gain each step.
  - After every update, the error-state correction is injected multiplicatively
    into the nominal state (quaternion update, not a linear add), and the
    error state is reset to zero. This "inject-then-reset" pattern is exactly
    what OpenVINS/MSCKF-style filters do and is what makes the filter stable
    for real attitude dynamics, unlike a plain additive EKF on a quaternion.

What this is NOT: OpenVINS itself is a full Multi-State Constraint Kalman
Filter (MSCKF) with a sliding window of cloned camera poses and SLAM feature
states, fed by a visual feature tracker. Your original code has no camera
feature tracker -- it consumes already-processed measurements (barometer,
optical-flow velocity, geo-referenced position match). There is nothing to
"clone" or triangulate, so the MSCKF sliding window has nothing to operate
on. What we *can* and do adopt from OpenVINS is its IMU error-state
propagation model plus properly frame-aware measurement Jacobians, which is
the part of the architecture that actually applies here and is the most
advanced formulation possible for these inputs.

State (nominal, 16 values): p (3), v (3), q (4, body->nav, Hamilton [w,x,y,z]),
bg (3, gyro bias), ba (3, accel bias).
Error state (15): dp, dv, dtheta, dbg, dba.
Frame: NED (z positive down), consistent with the original module.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.80665
GRAVITY_VEC = np.array([0.0, 0.0, GRAVITY])  # NED: gravity points +z (down)

# Error-state layout
IDX_P, IDX_V, IDX_TH, IDX_BG, IDX_BA = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
STATE_DIM = 15


# --------------------------------------------------------------------------- #
# Quaternion / rotation helpers (Hamilton convention, q = [w, x, y, z])
# --------------------------------------------------------------------------- #

def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric ("cross product") matrix of a 3-vector."""
    v = v.flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Body->nav rotation matrix from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_rotvec(theta: np.ndarray) -> np.ndarray:
    """Exact (not small-angle-only) quaternion for a rotation vector."""
    theta = theta.flatten()
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        return quat_normalize(np.array([1.0, *(theta / 2.0)]))
    axis = theta / angle
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])


# --------------------------------------------------------------------------- #
# Config helpers (same nested/flat fallback pattern as the original module)
# --------------------------------------------------------------------------- #

def _noise(config: dict, nested_key: str, flat_key: str, default: float) -> float:
    nested = config.get("measurement_noise", {})
    if nested_key in nested:
        return float(nested[nested_key])
    return float(config.get(flat_key, default))


def _imu_noise(config: dict, key: str, flat_key: str, default: float) -> float:
    nested = config.get("imu_noise", {})
    if key in nested:
        return float(nested[key])
    return float(config.get(flat_key, default))


class JamBoyEKF:
    """
    15-state Error-State EKF for GPS-denied navigation.
    Nominal state: p (NED m), v (NED m/s), q (body->nav attitude), gyro/accel bias.
    """

    def __init__(self, config: dict, baro_config: dict | None = None):
        baro_config = baro_config or {}

        # Nominal state
        self.p = np.zeros((3, 1))
        self.v = np.zeros((3, 1))
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # identity attitude
        self.bg = np.zeros((3, 1))
        self.ba = np.zeros((3, 1))

        # Error-state covariance (15x15)
        self.P = np.eye(STATE_DIM) * 1.0

        # IMU noise densities -> drive process noise during predict()
        self.sigma_gyro = _imu_noise(config, "gyro", "gyro_noise_density", 0.01)
        self.sigma_accel = _imu_noise(config, "accel", "accel_noise_density", 0.10)
        self.sigma_gyro_bias = _imu_noise(config, "gyro_bias", "gyro_bias_rw", 0.001)
        self.sigma_accel_bias = _imu_noise(config, "accel_bias", "accel_bias_rw", 0.01)

        # Measurement noise (unchanged semantics from the original module)
        self.r_baro = float(
            baro_config.get("noise_variance")
            or baro_config.get("meas_noise_m", 2.0) ** 2
            or config.get("meas_noise_baro", 4.0)
        )
        self.r_flow = _noise(config, "optical_flow", "meas_noise_flow", 2.0)
        self.r_geo = _noise(config, "geo_match", "meas_noise_geo_base", 5.0)
        self.gate_threshold = float(config.get("innovation_gate", 9.0))

        self._last_reject_reason: str | None = None

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def position_uncertainty_m(self) -> float:
        return float(np.sqrt(max(np.trace(self.P[IDX_P, IDX_P]), 0.0)))

    @property
    def confidence_score(self) -> float:
        """0-100% trust score derived from position covariance (lower P -> higher score)."""
        trace = float(np.trace(self.P[IDX_P, IDX_P]))
        return float(np.clip(100.0 * (1.0 / (1.0 + trace / 10.0)), 0.0, 100.0))

    # ------------------------------------------------------------------ #
    # Propagation (nonlinear nominal state + linear error-state covariance)
    # ------------------------------------------------------------------ #

    def predict(self, dt: float, gyro: np.ndarray | None = None, imu_accel: np.ndarray | None = None) -> None:
        """
        Propagate the filter by dt seconds using IMU measurements.

        gyro:      body-frame angular rate (rad/s). If None, attitude is held
                   fixed -- this reproduces the old filter's "no rotation"
                   assumption, so existing callers that only pass accel keep working.
        imu_accel: body-frame specific force (m/s^2), i.e. raw accelerometer output.
        """
        if dt <= 0:
            return

        omega = np.zeros(3) if gyro is None else np.asarray(gyro, dtype=float).flatten()
        accel = np.zeros(3) if imu_accel is None else np.asarray(imu_accel, dtype=float).flatten()

        omega_unbiased = omega - self.bg.flatten()
        accel_unbiased = accel - self.ba.flatten()

        R_old = quat_to_rot(self.q)
        a_nav = R_old @ accel_unbiased + GRAVITY_VEC  # rotate specific force into nav frame, add gravity

        # --- nominal state propagation (nonlinear) ---
        self.p = self.p + self.v * dt + 0.5 * a_nav.reshape(3, 1) * dt ** 2
        self.v = self.v + a_nav.reshape(3, 1) * dt
        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec(omega_unbiased * dt)))

        # --- error-state transition matrix F (first-order discretization) ---
        F = np.eye(STATE_DIM)
        F[IDX_P, IDX_V] = np.eye(3) * dt
        F[IDX_V, IDX_TH] = -R_old @ skew(accel_unbiased) * dt
        F[IDX_V, IDX_BA] = -R_old * dt
        F[IDX_TH, IDX_TH] = np.eye(3) - skew(omega_unbiased) * dt
        F[IDX_TH, IDX_BG] = -np.eye(3) * dt

        # --- noise mapping G and continuous PSD Qc -> discrete Qd ---
        G = np.zeros((STATE_DIM, 12))
        G[IDX_V, 3:6] = -R_old          # accel noise -> velocity error
        G[IDX_TH, 0:3] = -np.eye(3)     # gyro noise -> attitude error
        G[IDX_BG, 6:9] = np.eye(3)      # gyro bias random walk
        G[IDX_BA, 9:12] = np.eye(3)     # accel bias random walk

        Qc = np.diag([
            self.sigma_gyro ** 2, self.sigma_gyro ** 2, self.sigma_gyro ** 2,
            self.sigma_accel ** 2, self.sigma_accel ** 2, self.sigma_accel ** 2,
            self.sigma_gyro_bias ** 2, self.sigma_gyro_bias ** 2, self.sigma_gyro_bias ** 2,
            self.sigma_accel_bias ** 2, self.sigma_accel_bias ** 2, self.sigma_accel_bias ** 2,
        ])
        Qd = G @ Qc @ G.T * dt

        self.P = F @ self.P @ F.T + Qd

    # ------------------------------------------------------------------ #
    # Measurement updates
    # ------------------------------------------------------------------ #

    def update_barometer(self, z_baro: float) -> bool:
        H = np.zeros((1, STATE_DIM))
        H[0, 2] = 1.0  # dp_z
        h = np.array([[self.p[2, 0]]])
        R = np.array([[self.r_baro]])
        return self._update(np.array([[z_baro]]), h, H, R)

    def update_geo_match(self, z_geo: np.ndarray, confidence: float = 1.0) -> bool:
        H = np.zeros((2, STATE_DIM))
        H[0, 0] = 1.0  # dp_x
        H[1, 1] = 1.0  # dp_y
        h = self.p[0:2, :]
        noise = self.r_geo / max(float(confidence), 0.1)
        R = np.eye(2) * noise
        z = np.asarray(z_geo, dtype=float).reshape(2, 1)
        return self._update(z, h, H, R)

    def update_optical_flow(self, vx: float, vy: float, vz: float = 0.0, measure_vz: bool = False) -> bool:
        """
        Optical flow measures velocity in the BODY frame (it's a camera bolted
        to the vehicle), not the nav frame. The old linear filter fed flow
        straight into nav-frame velocity, which is only correct if the
        vehicle never rotates. Here we predict the body-frame velocity from
        the nominal state and take the Jacobian w.r.t. both velocity error
        AND attitude error, since rotating the state changes the predicted
        body-frame velocity too.
        """
        R = quat_to_rot(self.q)
        v_body_pred = R.T @ self.v  # (3,1)
        rows = 3 if measure_vz else 2

        H = np.zeros((rows, STATE_DIM))
        H[:, IDX_V] = (R.T)[:rows, :]
        H[:, IDX_TH] = skew(v_body_pred.flatten())[:rows, :]

        h = v_body_pred[:rows, :]
        z_full = np.array([[vx], [vy], [vz]])
        z = z_full[:rows, :]
        R_meas = np.eye(rows) * self.r_flow
        return self._update(z, h, H, R_meas)

    def zero_velocity_update(self) -> bool:
        """Zero-velocity update (ZUPT) as a genuine pseudo-measurement (body
        velocity == 0), rather than a hard state/covariance override."""
        return self.update_optical_flow(0.0, 0.0, 0.0, measure_vz=True)

    # ------------------------------------------------------------------ #
    # Core gated update + error-state injection
    # ------------------------------------------------------------------ #

    def _update(self, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        y = z - h
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
            mahal = float((y.T @ S_inv @ y).item()) if y.size == 1 else float((y.T @ S_inv @ y).trace())
        except np.linalg.LinAlgError:
            self._last_reject_reason = "singular_S"
            return False

        if mahal > self.gate_threshold:
            self._last_reject_reason = f"mahalanobis={mahal:.2f}"
            return False

        K = self.P @ H.T @ S_inv
        dx = K @ y  # (15,1) error-state correction

        # Inject error state into the nominal state (multiplicative for attitude)
        self.p = self.p + dx[IDX_P, :]
        self.v = self.v + dx[IDX_V, :]
        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec(dx[IDX_TH, :].flatten())))
        self.bg = self.bg + dx[IDX_BG, :]
        self.ba = self.ba + dx[IDX_BA, :]

        # Reset error state to zero; Joseph-form covariance update for numerical stability
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._last_reject_reason = None
        return True

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    def get_state(self) -> np.ndarray:
        """Position + velocity, matching the original 6-vector shape."""
        return np.concatenate([self.p.flatten(), self.v.flatten()])

    def get_full_state(self) -> dict:
        """Full 16-value nominal state, for callers that want attitude/bias too."""
        return {
            "position": self.p.flatten().copy(),
            "velocity": self.v.flatten().copy(),
            "quaternion_wxyz": self.q.copy(),
            "gyro_bias": self.bg.flatten().copy(),
            "accel_bias": self.ba.flatten().copy(),
        }


class _FilterPyCompat:
    """
    Shim so legacy code/tests can access `.ekf.x` as a flat vector and `.ekf.P`
    directly, matching the pre-upgrade interface.

    NOTE (breaking change): `.x` here is position+velocity only (length 6),
    since a quaternion doesn't have a meaningful flat additive form. `.P` is
    now 15x15 (was 6x6) because of the added attitude/bias error states --
    any code that indexed .P by raw position must be updated (P[0:3,0:3] is
    still position covariance, so most callers are unaffected).
    """

    def __init__(self, core: JamBoyEKF):
        self._core = core

    @property
    def x(self) -> np.ndarray:
        return self._core.get_state()

    @x.setter
    def x(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).flatten()
        self._core.p = value[0:3].reshape(3, 1)
        self._core.v = value[3:6].reshape(3, 1)

    @property
    def P(self) -> np.ndarray:
        return self._core.P


class NavigationEKF:
    """Backward-compatible facade over JamBoyEKF."""

    def __init__(self, config: dict, baro_config: dict | None = None):
        self._core = JamBoyEKF(config, baro_config=baro_config)

    @property
    def ekf(self) -> _FilterPyCompat:
        return _FilterPyCompat(self._core)

    @property
    def position(self) -> np.ndarray:
        return self._core.p.flatten()

    @property
    def velocity(self) -> np.ndarray:
        return self._core.v.flatten()

    @property
    def attitude_quaternion(self) -> np.ndarray:
        """New: [w, x, y, z] body->nav attitude. Not present in the old filter."""
        return self._core.q.copy()

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self._core.P))

    @property
    def confidence_score(self) -> float:
        return self._core.confidence_score

    def predict(self, dt: float, accel: np.ndarray | None = None, gyro: np.ndarray | None = None) -> None:
        """`accel` stays positional/2nd-arg for drop-in compatibility with old
        call sites. `gyro` is new -- pass it if you have it; attitude simply
        won't update if you don't, matching old behavior."""
        self._core.predict(dt, gyro=gyro, imu_accel=accel)

    def update_velocity(self, vx: float, vy: float, vz: float = 0.0) -> bool:
        return self._core.update_optical_flow(vx, vy, vz)

    def update_altitude(self, altitude_m: float) -> bool:
        return self._core.update_barometer(altitude_m)

    def update_position(self, x: float, y: float, confidence: float = 1.0) -> bool:
        return self._core.update_geo_match(np.array([x, y]), confidence=confidence)

    def set_position(self, x: float, y: float, z: float = 0.0) -> None:
        self._core.p = np.array([[x], [y], [z]])
        self._core.P[0, 0] = 1.0
        self._core.P[1, 1] = 1.0
        self._core.P[2, 2] = 1.0

    def set_velocity(self, vx: float, vy: float, vz: float = 0.0) -> None:
        self._core.v = np.array([[vx], [vy], [vz]])

    def zero_velocity(self) -> None:
        """Now a real ZUPT measurement update instead of a hard override."""
        self._core.zero_velocity_update()

    def get_state(self) -> np.ndarray:
        return self._core.get_state()
