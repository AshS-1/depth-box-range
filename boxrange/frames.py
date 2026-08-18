"""Frame sources: live RealSense, recorded replay, and a synthetic rig.

Every source yields the same :class:`Frame`, so the rest of the pipeline never
learns whether it is looking at a real camera. That matters for two reasons:
``pyrealsense2`` ships no wheel for several platforms (macOS arm64 among them),
and hardware-in-the-loop is a miserable way to test metric accuracy because you
have no ground truth. :class:`SyntheticSource` renders a box with a physically
motivated stereo noise model and *reports the true pose*, which is what the
accuracy tests assert against.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .geometry import OrientedBox, Plane
from .intrinsics import CameraIntrinsics, d435_depth, gemini335_depth


@dataclass(frozen=True)
class Frame:
    """One synchronised capture.

    ``depth_m`` is metres as float32 with 0 marking "no return" -- the RealSense
    invalid marker, kept rather than NaN so the array stays cheap to mask.
    """

    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    color: np.ndarray | None = None
    index: int = 0
    timestamp_s: float = 0.0

    @property
    def valid_fraction(self) -> float:
        return float((self.depth_m > 0).mean())


class FrameSource(Protocol):
    """Anything the pipeline can pull frames from."""

    def __iter__(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Live / recorded RealSense
# --------------------------------------------------------------------------


class RealSenseSource:
    """Live D400-series capture, or playback of a recorded ``.bag``.

    Depth is aligned to colour when colour is enabled, so a detection in the
    depth image indexes the same pixel in the RGB image -- required for any
    learned detector to hand a region back to the depth stage.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color: bool = True,
        bag_path: str | Path | None = None,
        warmup_frames: int = 10,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - depends on platform
            raise ImportError(
                "pyrealsense2 is required for RealSenseSource. Install with "
                "`pip install pyrealsense2` (Linux/Windows x86_64); on macOS "
                "arm64 build librealsense from source, or record a .bag and "
                "replay it with NpzSource/bag_path."
            ) from exc

        self._rs = rs
        self._closed = False
        self._pipeline = rs.pipeline()
        cfg = rs.config()

        if bag_path is not None:
            cfg.enable_device_from_file(str(bag_path), repeat_playback=False)
        else:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if color:
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = self._pipeline.start(cfg)

        if bag_path is not None:
            # Without this, playback drops frames to keep wall-clock pace and a
            # batch analysis silently skips most of the recording.
            profile.get_device().as_playback().set_real_time(False)

        # A recording may simply not contain colour, and asking for its stream
        # profile then throws. Detect what is actually present rather than
        # trusting the request.
        has_color = color and any(
            s.stream_type() == rs.stream.color for s in profile.get_streams()
        )
        if color and not has_color:
            color = False

        self._align = rs.align(rs.stream.color) if has_color else None
        self._want_color = has_color

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        # After alignment the depth image lives in the colour frame, so it is
        # the colour intrinsics that deproject it correctly.
        stream = rs.stream.color if has_color else rs.stream.depth
        vsp = profile.get_stream(stream).as_video_stream_profile()
        self.intrinsics = CameraIntrinsics.from_realsense(
            vsp.get_intrinsics(), depth_scale=depth_scale
        )

        baseline = self._read_baseline(profile)
        if baseline is not None and baseline > 0:
            self.intrinsics = replace(self.intrinsics, baseline_m=baseline)

        for _ in range(warmup_frames):
            # Auto-exposure needs a few frames to settle; depth from the first
            # frames is measurably worse.
            try:
                self._pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                break

        self._index = 0

    def _read_baseline(self, profile) -> float | None:
        """Actual stereo baseline from the device, for the noise model."""
        rs = self._rs
        try:
            left = profile.get_stream(rs.stream.infrared, 1)
            right = profile.get_stream(rs.stream.infrared, 2)
            t = left.get_extrinsics_to(right).translation
            return float(np.linalg.norm(np.asarray(t, dtype=np.float64)))
        except RuntimeError:
            return None  # Not all configs expose IR; the default is close enough.

    def __iter__(self) -> Iterator[Frame]:
        while not self._closed:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                return  # Timeout or end of a .bag file.

            if self._align is not None:
                frames = self._align.process(frames)

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            raw = np.asanyarray(depth_frame.get_data())
            depth_m = raw.astype(np.float32) * self.intrinsics.depth_scale

            color = None
            if self._want_color:
                cf = frames.get_color_frame()
                if cf:
                    color = np.asanyarray(cf.get_data()).copy()

            yield Frame(
                depth_m=depth_m,
                intrinsics=self.intrinsics,
                color=color,
                index=self._index,
                timestamp_s=float(depth_frame.get_timestamp()) / 1000.0,
            )
            self._index += 1

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            # Already stopped, or the device was unplugged mid-stream. Teardown
            # must not raise, or the `with` block masks the real error.
            with contextlib.suppress(RuntimeError):
                self._pipeline.stop()

    def __enter__(self) -> RealSenseSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Live Orbbec (AgiBot X2 chest camera)
# --------------------------------------------------------------------------


class OrbbecSource:
    """Live capture from an Orbbec Gemini 330-series camera.

    This is the AgiBot X2's chest RGB-D sensor (Gemini 335: depth 1280x800 @
    30 fps, 90 deg x 65 deg, 50 mm baseline, optimal range 0.26-3 m). It yields
    the same :class:`Frame` as every other source, so nothing downstream changes.

    Two differences from :class:`RealSenseSource` are worth knowing, because
    both fail quietly rather than loudly:

    **Depth scale is in millimetres.** ``depth_frame.get_depth_scale()`` returns
    millimetres per depth unit; the identically named RealSense call returns
    *metres* per unit. Treating one as the other is a factor of 1000, and since
    every threshold in this package scales with the noise model, the symptom is
    an empty detection list rather than an absurd distance. The conversion lives
    in :meth:`CameraIntrinsics.from_orbbec` and nowhere else.

    **Intrinsics come from the stream profile, not the camera param.**
    ``Pipeline.get_camera_param()`` returns a matched depth/colour pair and needs
    both sensors streaming, so it is unavailable in the depth-only mode this
    package recommends. ``profile.get_intrinsic()`` works with depth alone.
    """

    def __init__(
        self,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        color: bool = False,
        warmup_frames: int = 10,
        timeout_ms: int = 5000,
    ) -> None:
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:  # pragma: no cover - depends on platform
            raise ImportError(
                "pyorbbecsdk is required for OrbbecSource. Install with "
                "`pip install pyorbbecsdk` (Linux x86_64/aarch64 and Windows "
                "have wheels; macOS needs a source build). On the X2 itself the "
                "SDK is already present. To work without the camera, record on "
                "the robot with --record and replay the .npz anywhere."
            ) from exc

        self._ob = ob
        self._closed = False
        self._timeout_ms = int(timeout_ms)
        self._pipeline = ob.Pipeline()
        config = ob.Config()

        depth_profile = self._pick_profile(ob.OBSensorType.DEPTH_SENSOR, width, height, fps)
        config.enable_stream(depth_profile)

        self._align = None
        self._want_color = False
        if color:
            # Colour is off by default and should stay off for ranging. Aligning
            # depth into the colour frame trims the FOV from 90x65 to the colour
            # camera's 86x55 and resamples depth exactly at the discontinuities
            # segmentation keys on. Nothing here measures with colour.
            try:
                color_profile = self._pick_profile(
                    ob.OBSensorType.COLOR_SENSOR, width, height, fps, fmt=ob.OBFormat.RGB
                )
                config.enable_stream(color_profile)
                self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
                self._want_color = True
            except Exception:
                # A unit with no colour sensor, or one already claimed by another
                # process. Depth alone is the mode that matters, so carry on.
                self._want_color = False

        self._pipeline.start(config)
        self.intrinsics = self._read_intrinsics(depth_profile)

        for _ in range(warmup_frames):
            # The first frames land before auto-exposure settles and are
            # measurably noisier.
            if self._pipeline.wait_for_frames(self._timeout_ms) is None:
                break

        self._index = 0

    def _pick_profile(self, sensor_type, width: int, height: int, fps: int, fmt=None):
        """Requested profile, or the device default if that exact mode is absent.

        Zero means "any" to the SDK. Falling back rather than raising matters
        because the same code runs against a Gemini 335 on the robot and
        whatever development unit is on the bench, and those support different
        mode tables.
        """
        ob = self._ob
        profiles = self._pipeline.get_stream_profile_list(sensor_type)
        if width or height or fps or fmt is not None:
            default_fmt = ob.OBFormat.Y16 if sensor_type == ob.OBSensorType.DEPTH_SENSOR else None
            try:
                return profiles.get_video_stream_profile(
                    width, height, fmt if fmt is not None else default_fmt, fps
                )
            except ob.OBError:
                pass
        return profiles.get_default_video_stream_profile()

    def _read_intrinsics(self, depth_profile) -> CameraIntrinsics:
        """Per-device calibration, with the depth scale folded in.

        The scale is a property of a *frame*, not a profile, so one frame has to
        be pulled before the intrinsics are complete. A camera that never
        delivers one is a hard failure -- guessing 1 mm/unit here would produce
        plausible-looking distances that are silently wrong on any device
        configured for 0.1 mm units.
        """
        frames = None
        for _ in range(10):
            frames = self._pipeline.wait_for_frames(self._timeout_ms)
            if frames is not None and frames.get_depth_frame() is not None:
                break
            frames = None
        if frames is None:
            self.close()
            raise RuntimeError(
                "Orbbec camera started but delivered no depth frame. Check the "
                "USB 3 cable and, on Linux, that the udev rules from the SDK are "
                "installed (99-obsensor-libusb.rules)."
            )

        depth_frame = frames.get_depth_frame()
        intr = CameraIntrinsics.from_orbbec(
            depth_profile.get_intrinsic(), depth_frame.get_depth_scale()
        )
        # Gemini 330-series baseline, for the z^2 noise model. The SDK exposes no
        # baseline query, so it comes from the datasheet rather than the device.
        return replace(intr, baseline_m=0.050, subpixel_px=gemini335_depth().subpixel_px)

    @staticmethod
    def _timestamp_s(frame) -> float:
        """Frame timestamp in seconds, whatever the SDK version calls it.

        pyorbbecsdk renamed this between v1 and v2 and the units differ by
        accessor. Wrong-but-monotonic timestamps only affect logging here, so a
        miss falls back to zero instead of raising.
        """
        for name, scale in (("get_timestamp_us", 1e-6), ("get_timestamp", 1e-3)):
            getter = getattr(frame, name, None)
            if getter is not None:
                with contextlib.suppress(Exception):
                    return float(getter()) * scale
        return 0.0

    def __iter__(self) -> Iterator[Frame]:
        ob = self._ob
        while not self._closed:
            frames = self._pipeline.wait_for_frames(self._timeout_ms)
            if frames is None:
                continue
            if self._align is not None:
                frames = self._align.process(frames)
                if frames is None:
                    continue
                frames = frames.as_frame_set()

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            h, w = depth_frame.get_height(), depth_frame.get_width()
            raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
            depth_m = raw.astype(np.float32) * self.intrinsics.depth_scale

            color = None
            if self._want_color:
                cf = frames.get_color_frame()
                if cf is not None and cf.get_format() == ob.OBFormat.RGB:
                    rgb = np.frombuffer(cf.get_data(), dtype=np.uint8).reshape(
                        cf.get_height(), cf.get_width(), 3
                    )
                    color = rgb[..., ::-1].copy()  # the overlay draws in BGR

            yield Frame(
                depth_m=depth_m,
                intrinsics=self.intrinsics,
                color=color,
                index=self._index,
                timestamp_s=self._timestamp_s(depth_frame),
            )
            self._index += 1

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            # Teardown must not raise or a `with` block masks the real error.
            with contextlib.suppress(Exception):
                self._pipeline.stop()

    def __enter__(self) -> OrbbecSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Recorded arrays
# --------------------------------------------------------------------------


class NpzSource:
    """Replay frames captured with :func:`record_npz`.

    The point is to record on the machine with the camera and analyse anywhere.
    """

    def __init__(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        depth = data["depth_m"]
        # Tolerate a single frame saved as (H, W) rather than (1, H, W).
        self._depth = depth[None] if depth.ndim == 2 else depth
        self._color = data["color"] if "color" in data.files else None
        self.intrinsics = CameraIntrinsics(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            depth_scale=float(data["depth_scale"]) if "depth_scale" in data.files else 1e-3,
            baseline_m=float(data["baseline_m"]) if "baseline_m" in data.files else 0.050,
        )

    def __iter__(self) -> Iterator[Frame]:
        for i, depth in enumerate(self._depth):
            yield Frame(
                depth_m=depth.astype(np.float32),
                intrinsics=self.intrinsics,
                color=None if self._color is None else self._color[i],
                index=i,
                timestamp_s=i / 30.0,
            )

    def close(self) -> None:
        pass

    def __enter__(self) -> NpzSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def record_npz(source: FrameSource, path: str | Path, max_frames: int = 150) -> int:
    """Drain ``source`` into an ``.npz`` that :class:`NpzSource` can replay."""
    depths, colors, intr = [], [], None
    for frame in source:
        intr = frame.intrinsics
        depths.append(frame.depth_m.astype(np.float32))
        colors.append(frame.color)
        if len(depths) >= max_frames:
            break

    if intr is None:
        raise RuntimeError("source produced no frames")

    # Colour is saved only if *every* frame had it. Appending only the frames
    # that did would silently shift colour out of step with depth, so replay
    # would pair each depth frame with some other frame's image.
    if any(c is None for c in colors):
        colors = []

    payload = {
        "depth_m": np.asarray(depths, dtype=np.float32),
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        "depth_scale": intr.depth_scale, "baseline_m": intr.baseline_m,
    }
    if colors:
        payload["color"] = np.asarray(colors, dtype=np.uint8)
    np.savez_compressed(Path(path), **payload)
    return len(depths)


# --------------------------------------------------------------------------
# Synthetic rig
# --------------------------------------------------------------------------


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass(frozen=True)
class SyntheticScene:
    """A box standing on a floor, described in camera-relative terms.

    Angles are radians. ``forward_m`` is measured along the floor from the point
    directly beneath the camera, which is how you would actually place a box
    with a tape measure.
    """

    box_size: tuple[float, float, float] = (0.40, 0.30, 0.25)
    forward_m: float = 1.60
    lateral_m: float = 0.0
    yaw: float = np.deg2rad(25.0)
    camera_height_m: float = 0.90
    camera_pitch: float = np.deg2rad(18.0)

    def ground_plane(self) -> Plane:
        """The floor in the depth optical frame (+x right, +y down, +z fwd)."""
        # Pitching the camera down by theta rotates world-up in the camera frame
        # about the camera x axis.
        th = self.camera_pitch
        up = np.array([0.0, -np.cos(th), -np.sin(th)])
        return Plane(up, float(self.camera_height_m))

    def truth_box(self) -> OrientedBox:
        plane = self.ground_plane()
        n = plane.normal
        # Floor point directly below the camera.
        origin = plane.point

        z_axis = np.array([0.0, 0.0, 1.0])
        fwd = z_axis - (z_axis @ n) * n
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, n)

        foot = origin + self.forward_m * fwd + self.lateral_m * right
        rot = _rodrigues(n, self.yaw)
        a1, a2 = rot @ fwd, rot @ right

        d, w, h = self.box_size
        center = foot + (h / 2.0) * n
        R = np.stack((a1, a2, n), axis=1)
        if np.linalg.det(R) < 0:
            R[:, 1] = -R[:, 1]
        return OrientedBox(center, R, np.array([d, w, h], dtype=np.float64))


def render_depth(
    scene: SyntheticScene,
    intr: CameraIntrinsics,
    *,
    noise: bool = True,
    dropout: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ray-cast the scene into a depth image, in metres.

    Rays are built with a z-component of exactly 1, so the ray parameter *is*
    the depth -- no normalising, no division later.

    Noise is injected in the disparity domain, not the depth domain. That is
    where a stereo camera's error actually lives, and it is what makes far
    returns noisier than near ones by z^2, matching
    :meth:`CameraIntrinsics.range_sigma`.
    """
    rng = rng or np.random.default_rng(0)
    h, w = intr.height, intr.width
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dirs = np.stack(
        ((uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu)), axis=-1
    )
    flat = dirs.reshape(-1, 3)

    # Floor: n . (t*d) + dd = 0
    plane = scene.ground_plane()
    denom = flat @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = np.where(np.abs(denom) > 1e-9, -plane.d / denom, np.inf)
    t_plane = np.where(t_plane > 0, t_plane, np.inf)

    # Box: slab test in the box's own frame.
    box = scene.truth_box()
    o = -box.R.T @ box.center
    d_local = flat @ box.R
    half = box.extents / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d_local
        t2 = (half - o) / d_local
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 0)
    t_box = np.where(hit, np.where(t_near > 0, t_near, t_far), np.inf)

    z = np.minimum(t_plane, t_box)
    z[~np.isfinite(z)] = 0.0
    z = z.reshape(h, w)

    if noise:
        m = z > 0
        # depth -> disparity -> perturb -> back
        disp = np.zeros_like(z)
        disp[m] = intr.fx * intr.baseline_m / z[m]
        disp[m] += rng.normal(0.0, intr.subpixel_px, size=int(m.sum()))
        z = np.zeros_like(z)
        good = m & (disp > 1e-6)
        z[good] = intr.fx * intr.baseline_m / disp[good]
        # The sensor reports integer depth units, and that quantisation is a
        # real error floor at close range.
        z = np.round(z / intr.depth_scale) * intr.depth_scale

    if dropout > 0:
        z[rng.random(z.shape) < dropout] = 0.0

    return z.astype(np.float32)


class SyntheticSource:
    """Renders ``frames`` images of a scene, with fresh noise each frame."""

    def __init__(
        self,
        scene: SyntheticScene | None = None,
        intrinsics: CameraIntrinsics | None = None,
        *,
        frames: int = 60,
        noise: bool = True,
        seed: int = 0,
    ) -> None:
        self.scene = scene or SyntheticScene()
        # One definition of "a D435 depth stream", shared with the presets, so
        # the selftest table and anything built on this default cannot drift
        # apart by half a pixel and quietly report different accuracies.
        self.intrinsics = intrinsics or d435_depth()
        self._frames = frames
        self._noise = noise
        self._rng = np.random.default_rng(seed)

    def truth_box(self) -> OrientedBox:
        return self.scene.truth_box()

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self._frames):
            yield Frame(
                depth_m=render_depth(
                    self.scene, self.intrinsics, noise=self._noise, rng=self._rng
                ),
                intrinsics=self.intrinsics,
                index=i,
                timestamp_s=i / 30.0,
            )

    def close(self) -> None:
        pass

    def __enter__(self) -> SyntheticSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
