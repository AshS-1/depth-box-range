"""Overlay rendering, for eyeballing whether the fit is sane."""

from __future__ import annotations

import cv2
import numpy as np

from .intrinsics import CameraIntrinsics
from .pipeline import BoxDetection

# Corner ordering matches OrientedBox.corners(): bottom face 0-3, top face 4-7.
_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _to_pixel(uv: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int] | None:
    """Round a projected point to an int pair OpenCV will accept.

    A corner near the image plane (tiny positive z) projects to coordinates in
    the 1e13 range. Those are finite, so a plain isfinite() check passes them
    straight into cv2.line, which then raises on int32 overflow -- a real crash
    in --show whenever a box drifts near the frame edge. Clamping to a bound
    well outside the image keeps the on-screen part of the line identical while
    staying inside int32.
    """
    if not np.isfinite(uv).all():
        return None
    bound = 10 * max(shape[0], shape[1])
    x, y = np.clip(uv, -bound, bound)
    return round(float(x)), round(float(y))


def project(points: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixels. Points behind go to NaN."""
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    safe = np.where(np.abs(z) > 1e-6, z, np.nan)
    u = pts[:, 0] / safe * intr.fx + intr.cx
    v = pts[:, 1] / safe * intr.fy + intr.cy
    uv = np.stack((u, v), axis=1)
    uv[z <= 1e-6] = np.nan
    return uv


def colorize_depth(depth_m: np.ndarray, max_range_m: float = 5.0) -> np.ndarray:
    """Depth to a BGR image. Invalid pixels render black, not as near returns."""
    valid = depth_m > 0
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        scaled = np.clip(depth_m / max_range_m, 0.0, 1.0)
        norm[valid] = (255 * (1.0 - scaled[valid])).astype(np.uint8)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def draw_detection(
    image: np.ndarray,
    det: BoxDetection,
    intr: CameraIntrinsics,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    show_mask: bool = True,
) -> np.ndarray:
    """Draw the wireframe, the nearest-point marker, and the distance label."""
    out = image
    h, w = out.shape[:2]

    if show_mask and det.candidate.mask.shape == (h, w):
        tint = out.copy()
        tint[det.candidate.mask] = color
        out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)

    uv = project(det.box.corners(), intr)
    for a, b in _EDGES:
        pa, pb = _to_pixel(uv[a], out.shape), _to_pixel(uv[b], out.shape)
        if pa is None or pb is None:
            continue
        cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)

    # Mark the point the headline distance actually refers to.
    from .ranging import closest_point_on_box

    near = _to_pixel(project(closest_point_on_box(det.box)[None, :], intr)[0], out.shape)
    if near is not None:
        cv2.circle(out, near, 6, (0, 0, 255), -1, cv2.LINE_AA)

    finite = uv[np.isfinite(uv).all(axis=1)]
    anchor = (
        np.round(finite.min(axis=0)).astype(int) if len(finite) else np.array([10, 30])
    )
    r = det.range
    label = f"#{det.track_id} {r.surface_m:.2f}m +/-{r.sigma_m:.3f}"
    sub = f"conf {r.confidence:.2f} faces {r.visible_faces}"

    x = int(np.clip(anchor[0], 4, w - 220))
    y = int(np.clip(anchor[1] - 10, 24, h - 10))
    cv2.rectangle(out, (x - 4, y - 20), (x + 214, y + 16), (0, 0, 0), -1)
    cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(out, sub, (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_frame(
    depth_m: np.ndarray,
    detections: list[BoxDetection],
    intr: CameraIntrinsics,
    *,
    background: np.ndarray | None = None,
    max_range_m: float = 5.0,
) -> np.ndarray:
    """Full overlay: colourised depth (or the colour image) plus every box."""
    base = background.copy() if background is not None else colorize_depth(depth_m, max_range_m)
    for det in detections:
        base = draw_detection(base, det, intr)
    return base
