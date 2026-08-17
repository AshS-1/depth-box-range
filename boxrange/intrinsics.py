"""Pinhole camera intrinsics plus the stereo depth noise model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for a depth stream.

    ``depth_scale`` converts raw depth units to metres. RealSense D400 devices
    ship 16-bit depth in millimetres, so the default is 1e-3.

    ``baseline_m`` and ``subpixel_px`` describe the stereo rig and are only used
    by :meth:`range_sigma`. The defaults match a D435/D435i.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1e-3
    baseline_m: float = 0.050
    subpixel_px: float = 0.08

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(cls, K, width: int, height: int, **kw) -> CameraIntrinsics:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        return cls(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], **kw)

    @classmethod
    def from_realsense(cls, intr, depth_scale: float = 1e-3) -> CameraIntrinsics:
        """Build from a ``pyrealsense2.intrinsics`` object."""
        return cls(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            depth_scale=depth_scale,
        )

    def scaled(self, factor: float) -> CameraIntrinsics:
        """Intrinsics for an image resized by ``factor`` (0.5 = half size)."""
        return replace(
            self,
            width=round(self.width * factor),
            height=round(self.height * factor),
            fx=self.fx * factor,
            fy=self.fy * factor,
            # Pixel centres shift by half a pixel under the usual resize convention.
            cx=(self.cx + 0.5) * factor - 0.5,
            cy=(self.cy + 0.5) * factor - 0.5,
        )

    def range_sigma(self, z_m):
        """1-sigma depth uncertainty at range ``z_m``, in metres.

        Stereo triangulation error grows with the square of range:

            sigma_z = z^2 * sigma_disparity / (focal_px * baseline)

        This is the dominant error term beyond ~1 m and is why a single "nearest
        pixel" reading is a bad distance estimate at range -- the tail of the
        noise distribution reaches toward the camera.
        """
        z = np.asarray(z_m, dtype=np.float64)
        return z * z * self.subpixel_px / (self.fx * self.baseline_m)
