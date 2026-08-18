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
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        vfov_deg: float | None = None,
        **kw,
    ) -> CameraIntrinsics:
        """Nominal intrinsics from a datasheet field of view.

        For a stand-in when the camera is not plugged in -- simulating the X2's
        chest camera on a laptop, say. Always prefer the factory calibration
        that :meth:`from_orbbec` reads off the device.

        The image spans ``width`` pixels of *area*, so its half-extent is
        ``width / 2`` in continuous image coordinates while the centre pixel sits
        at ``(width - 1) / 2``. Mixing those two conventions is a half-pixel
        error, which is small, and using ``width / 2`` for both is the usual way
        to make it.

        ``vfov_deg`` defaults to None, which forces square pixels (``fy == fx``)
        and lets the vertical FOV follow from the aspect ratio. That is what real
        sensors do. Pass both only to reproduce a datasheet that quotes two
        independently rounded numbers.
        """
        if not 0.0 < hfov_deg < 180.0:
            raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        if vfov_deg is None:
            fy = fx
        else:
            if not 0.0 < vfov_deg < 180.0:
                raise ValueError(f"vfov_deg must be in (0, 180), got {vfov_deg}")
            fy = (height / 2.0) / np.tan(np.deg2rad(vfov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            **kw,
        )

    @classmethod
    def from_orbbec(cls, intr, depth_scale_mm: float) -> CameraIntrinsics:
        """Build from a ``pyorbbecsdk`` ``OBCameraIntrinsic``.

        ``depth_scale_mm`` is what ``depth_frame.get_depth_scale()`` returns, and
        it is **millimetres per depth unit** -- not metres. The RealSense call of
        the same name returns metres per unit. Passing an Orbbec scale straight
        into ``depth_scale`` puts every distance out by a factor of 1000, and
        because the pipeline's thresholds all scale with the noise model it
        degrades into an empty detection list rather than an obviously wrong
        number. Hence the separate constructor and the separate argument name.
        """
        return cls(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
            depth_scale=float(depth_scale_mm) * 1e-3,
        )

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


def subpixel_from_spec(
    *, rel_error: float, at_range_m: float, fx: float, baseline_m: float
) -> float:
    """Invert :meth:`CameraIntrinsics.range_sigma` to get the subpixel constant.

    Datasheets quote depth error as a percentage at a stated range; the noise
    model wants disparity noise in pixels. This converts one into the other so
    the number in a preset is traceable to a published figure instead of folklore.
    """
    sigma_m = rel_error * at_range_m
    return float(sigma_m * fx * baseline_m / (at_range_m**2))


def gemini335_depth(width: int = 1280, height: int = 800, **overrides) -> CameraIntrinsics:
    """Nominal depth intrinsics for the Orbbec Gemini 335 -- the AgiBot X2's
    chest RGB-D camera.

    Datasheet: depth FOV 90 deg x 65 deg, up to 1280x800 @ 30 fps, 50 mm
    baseline, range 0.10-20 m with an optimal band of 0.26-3 m, spatial
    precision <= 1.5% at 2 m.

    Only the horizontal FOV is used, which forces square pixels and puts the
    vertical FOV at 2*atan(400/640) = 64.0 deg. The datasheet says 65 deg +/- 3,
    so the two agree; the quoted pair is simply rounded independently and taking
    both at face value would imply non-square pixels the sensor does not have.

    ``subpixel_px`` is derived from the 1.5% at 2 m precision figure, which is a
    *bound* rather than a typical value, so :meth:`CameraIntrinsics.range_sigma`
    errs wide. That is the safe direction for obstacle avoidance but it also
    makes the confidence score generous -- measure your own unit against a flat
    wall and override it if you need the tighter number.

    This is a stand-in for bench work without hardware. :class:`OrbbecSource`
    reads the real per-device calibration off the camera and should be preferred
    whenever one is plugged in.
    """
    intr = CameraIntrinsics.from_fov(
        width, height, hfov_deg=90.0, depth_scale=1e-3, baseline_m=0.050
    )
    intr = replace(
        intr,
        subpixel_px=subpixel_from_spec(
            rel_error=0.015, at_range_m=2.0, fx=intr.fx, baseline_m=intr.baseline_m
        ),
    )
    return replace(intr, **overrides) if overrides else intr


def d435_depth(width: int = 640, height: int = 480, **overrides) -> CameraIntrinsics:
    """Nominal depth intrinsics for a RealSense D435/D435i at 640x480."""
    intr = CameraIntrinsics(
        width=width,
        height=height,
        fx=385.0 * width / 640.0,
        fy=385.0 * width / 640.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )
    return replace(intr, **overrides) if overrides else intr


CAMERA_PRESETS = {"gemini335": gemini335_depth, "d435": d435_depth}

# A sensible bench resolution per preset -- native aspect ratio, and small
# enough to stay interactive on a CPU. The Gemini 335's is 8:5, so forcing it to
# a 4:3 640x480 would model a 74 degree vertical FOV the real sensor does not
# have. Its full 1280x800 is native but four times the pixels.
PRESET_RESOLUTIONS = {"gemini335": (640, 400), "d435": (640, 480)}
