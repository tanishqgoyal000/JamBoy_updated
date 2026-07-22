"""
LoFTR geo-matching backend for JamBoy.

Wraps kornia's LoFTR model and provides loftr_match_tile(), a drop-in
replacement for ORB's feature-match step. Everything downstream (RANSAC,
altitude-scale gate, pixel->WGS84 conversion, confidence scoring) is reused
unchanged from GeoMatcher.
"""
from __future__ import annotations
import numpy as np
import cv2


class LoFTRBackend:
    """Wraps kornia LoFTR. One instance, reused for all tiles."""

    def __init__(self, device: str = "cpu", pretrained: str = "outdoor",
                 max_dim: int = 640, conf_threshold: float = 0.5):
        import torch
        from kornia.feature import LoFTR
        self.torch = torch
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = LoFTR(pretrained=pretrained).to(self.device).eval()
        self.max_dim = int(max_dim)
        self.conf_threshold = float(conf_threshold)

    def _prep(self, gray: np.ndarray):
        h, w = gray.shape[:2]
        scale = min(1.0, self.max_dim / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        t = self.torch.from_numpy(gray).float()[None, None] / 255.0
        return t.to(self.device), scale

    def match(self, live_gray: np.ndarray, tile_gray: np.ndarray):
        t0, s0 = self._prep(live_gray)
        t1, s1 = self._prep(tile_gray)
        with self.torch.no_grad():
            out = self.model({"image0": t0, "image1": t1})
        mk0 = out["keypoints0"].cpu().numpy()
        mk1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()
        keep = conf >= self.conf_threshold
        mk0, mk1, conf = mk0[keep], mk1[keep], conf[keep]
        if s0 != 1.0:
            mk0 = mk0 / s0
        if s1 != 1.0:
            mk1 = mk1 / s1
        return mk0, mk1, conf


def loftr_match_tile(self, live_gray, tile, altitude_m=None):
    """
    LoFTR replacement for _match_tile(). Expects to be bound onto a
    GeoMatcher instance (via .__get__(self)), so self._loftr,
    self.min_good_matches, self.ransac_threshold, etc. are all available.
    """
    try:
        mk_live, mk_tile, conf = self._loftr.match(live_gray, tile.image)
    except Exception:
        return None
    if len(mk_live) < max(4, int(self.min_good_matches * self.partial_match_ratio)):
        return None

    src_pts = mk_live.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = mk_tile.reshape(-1, 1, 2).astype(np.float32)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
    if M is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())
    good_count = len(mk_live)
    inlier_ratio = inliers / max(good_count, 1)
    if inlier_ratio < self.min_inlier_ratio:
        return None
    if not self._altitude_scale_ok(M, altitude_m):
        return None

    h, w = live_gray.shape
    center = np.float32([[[w / 2, h / 2]]])
    map_pixel = cv2.perspectiveTransform(center, M)
    px, py = map_pixel[0][0]

    lat, lon = self.map_index.pixel_to_wgs84(tile, float(px), float(py))
    x_local, y_local = self.map_index.wgs84_to_local_m(
        tile, lat, lon, self.origin_lat, self.origin_lon
    )

    is_partial = good_count < self.min_good_matches
    confidence = self._score_confidence(inlier_ratio, inliers, good_count, is_partial)
    if len(conf):
        confidence = float(np.clip(0.5 * confidence + 0.5 * float(conf.mean()), 0.0, 1.0))
    if confidence < self.min_confidence:
        return None

    from jamboy.geo_match import GeoFix
    return GeoFix(
        lat=lat, lon=lon, confidence=confidence,
        x_local=x_local, y_local=y_local,
        map_season=tile.season, match_mode="loftr",
    )
