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
import os
import time
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
# Live Orbbec, straight over USB
# --------------------------------------------------------------------------


class OrbbecSource:
    """Live capture from an Orbbec Gemini 330-series camera.

    The same part the AgiBot X2 carries in its head (Gemini 335: depth 1280x800
    @ 30 fps, 90 deg x 65 deg, 50 mm baseline, optimal range 0.26-3 m). It yields
    the same :class:`Frame` as every other source, so nothing downstream changes.

    **On the X2 itself, use :class:`X2RgbdSource` instead.** This class claims
    the USB device, and on the robot that device already has an owner. This one
    is for a Gemini 335 on a bench.

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
# AgiBot X2 head RGB-D, over the robot's own ROS 2 interface
# --------------------------------------------------------------------------

# From the X2 AIMDK sensor interface docs. Note "head_front": the RGB-D camera
# is in the head, not the chest -- the chest carries the LiDAR and an IMU.
X2_DEPTH_IMAGE = "/aima/hal/sensor/rgbd_head_front/depth_image"
X2_DEPTH_INFO = "/aima/hal/sensor/rgbd_head_front/depth_camera_info"
X2_RGB_IMAGE = "/aima/hal/sensor/rgbd_head_front/rgb_image"


def decode_depth_image(
    data: bytes, height: int, width: int, encoding: str, step: int, is_bigendian: bool = False
) -> np.ndarray:
    """A ``sensor_msgs/Image`` depth payload as metres, float32, 0 for no return.

    The X2 docs give the topic and the message type but not the encoding, and
    ROS publishes depth two different ways: ``16UC1`` in *millimetres* and
    ``32FC1`` in *metres*. Guessing costs a factor of 1000 in one direction, so
    this reads ``msg.encoding`` and refuses anything it does not recognise
    rather than defaulting.

    ``step`` is the row stride in bytes and is not always ``width * itemsize`` --
    rows can be padded. Reshaping by width alone silently shears the image when
    they are, which looks like a wildly tilted floor rather than like a bug.
    """
    if encoding in ("16UC1", "mono16"):
        # Millimetres. NaN is not representable, so 0 is the no-return marker,
        # which is also this package's convention.
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        scale = 1e-3
    elif encoding == "32FC1":
        # Metres already, but NaN and inf are, so fold them into the 0 marker.
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(
            f"unsupported depth encoding {encoding!r}; expected 16UC1 (millimetres) "
            "or 32FC1 (metres). Check the publisher with "
            f"`ros2 topic echo --field encoding {X2_DEPTH_IMAGE}`."
        )

    per_row = step // dtype.itemsize
    if per_row < width:
        raise ValueError(f"step {step} is too small for width {width} at {encoding}")

    raw = np.frombuffer(data, dtype=dtype, count=per_row * height)
    depth = raw.reshape(height, per_row)[:, :width].astype(np.float32) * scale
    with np.errstate(invalid="ignore"):
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
    return depth


def depth_quantisation(encoding: str) -> float:
    """Depth quantisation step in metres, for the uncertainty model.

    ``16UC1`` resolves to a millimetre and that floor is real at close range.
    ``32FC1`` carries far more precision than the sensor has, so it contributes
    nothing and would only inflate the reported sigma.
    """
    return 1e-3 if encoding in ("16UC1", "mono16") else 0.0


def intrinsics_from_camera_info(msg, **overrides) -> CameraIntrinsics:
    """Build from a ``sensor_msgs/CameraInfo``.

    This is the per-device factory calibration the robot publishes, so it beats
    the datasheet preset in :func:`~boxrange.intrinsics.gemini335_depth` and
    should always win where both are available.

    ``CameraInfo`` carries no stereo baseline, so the noise model still takes
    the Gemini 335's datasheet 50 mm.
    """
    # rclpy exposes the intrinsic matrix as `k`; ROS 1 and some bridges use `K`.
    k = getattr(msg, "k", None)
    if k is None:
        k = getattr(msg, "K", None)
    if k is None:
        raise ValueError("CameraInfo has neither `k` nor `K`")

    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(
            f"CameraInfo carries no usable intrinsics (fx={k[0, 0]}, fy={k[1, 1]}). "
            "The camera is publishing images before calibration is loaded."
        )

    preset = gemini335_depth()
    return replace(
        CameraIntrinsics(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            baseline_m=preset.baseline_m,
            subpixel_px=preset.subpixel_px,
        ),
        **overrides,
    )


class X2RgbdSource:
    """The AgiBot X2's head RGB-D camera, over the robot's ROS 2 interface.

    This is the supported way to get depth on the robot. :class:`OrbbecSource`
    opens the Gemini 335 directly over USB, which works on a bench but on the
    X2 means competing with the robot's own stack for the device -- a USB camera
    has one owner, and the loser gets "device busy" or nothing at all.

    Topics, from the AIMDK sensor interface docs, at 30 Hz::

        /aima/hal/sensor/rgbd_head_front/depth_image        sensor_msgs/Image
        /aima/hal/sensor/rgbd_head_front/depth_camera_info  sensor_msgs/CameraInfo

    Intrinsics come from ``depth_camera_info``, so this needs no datasheet
    guesswork about focal length -- only the stereo baseline, which CameraInfo
    does not carry.

    Subscriptions only: this creates no publishers, services, parameters or
    actions, so running it cannot perturb anything else using the robot.

    **QoS defaults to BEST_EFFORT**, and that is deliberate. A RELIABLE
    subscriber against a BEST_EFFORT publisher is an incompatible pair and
    receives *nothing*, with no error -- the single most common way a ROS 2
    camera subscription appears dead. BEST_EFFORT is compatible with publishers
    of either kind, and for a camera dropping a stale frame is what you want
    anyway. The docs say these topics publish RELIABLE, so pass
    ``reliable=True`` if you would rather have the matching pair and the
    delivery guarantee.
    """

    def __init__(
        self,
        *,
        depth_topic: str = X2_DEPTH_IMAGE,
        info_topic: str = X2_DEPTH_INFO,
        color_topic: str = X2_RGB_IMAGE,
        color: bool = False,
        reliable: bool = False,
        queue_depth: int = 1,
        timeout_s: float = 5.0,
        node_name: str | None = None,
    ) -> None:
        try:
            import rclpy
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:  # pragma: no cover - needs a ROS 2 install
            raise ImportError(
                "rclpy and sensor_msgs are required for X2RgbdSource. They come "
                "from a ROS 2 installation, not from pip -- source the robot's "
                "setup.bash (or your own ROS 2 Humble install) and run again. "
                "To work off-robot, record on the X2 with --record and replay "
                "the .npz anywhere."
            ) from exc

        self._rclpy = rclpy
        self._closed = False
        self._timeout_s = float(timeout_s)

        # Do not tear down a context this process did not create: boxrange may be
        # one node inside a larger application that called rclpy.init itself.
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        # Unique by default. The X2 is shared hardware, and two people running
        # this at once -- or one person leaving a stale session behind -- puts two
        # nodes with the same name on the graph, which ROS 2 warns about and
        # which makes `ros2 node list` ambiguous for everyone else. The pid is
        # enough to separate them and still says what the node is.
        self._node = rclpy.create_node(node_name or f"boxrange_x2_{os.getpid()}")
        qos = QoSProfile(
            depth=queue_depth,
            reliability=(
                QoSReliabilityPolicy.RELIABLE if reliable else QoSReliabilityPolicy.BEST_EFFORT
            ),
        )

        self._depth_msg = None
        self._color_msg = None
        self._info = None

        self._node.create_subscription(Image, depth_topic, self._on_depth, qos)
        self._node.create_subscription(CameraInfo, info_topic, self._on_info, qos)
        self._want_color = color
        if color:
            self._node.create_subscription(Image, color_topic, self._on_color, qos)

        self._depth_topic = depth_topic
        self._info_topic = info_topic
        self.intrinsics = self._await_intrinsics()
        self._index = 0

    # -- callbacks: keep only the newest, since ranging wants the latest state --

    def _on_depth(self, msg) -> None:
        self._depth_msg = msg

    def _on_color(self, msg) -> None:
        self._color_msg = msg

    def _on_info(self, msg) -> None:
        self._info = msg

    def _spin_until(self, predicate) -> bool:
        """Pump callbacks until ``predicate`` holds or the timeout expires.

        Spinning here rather than on a background thread keeps this a pull-based
        iterator like every other source, so the pipeline never learns it is
        talking to ROS and there is no shared state to lock.
        """
        deadline = time.monotonic() + self._timeout_s
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._rclpy.spin_once(self._node, timeout_sec=min(remaining, 0.1))
        return True

    def _await_intrinsics(self) -> CameraIntrinsics:
        """Block until CameraInfo arrives; without it there is no metric answer.

        Failing here with the topic name beats streaming frames that cannot be
        deprojected, and separates "nothing is publishing" from "the calibration
        has not been loaded yet", which look identical from a dead pipeline.
        """
        if not self._spin_until(lambda: self._info is not None and self._depth_msg is not None):
            missing = []
            if self._info is None:
                missing.append(self._info_topic)
            if self._depth_msg is None:
                missing.append(self._depth_topic)
            self.close()
            raise RuntimeError(
                f"no messages on {' and '.join(missing)} within {self._timeout_s:.0f}s. "
                "Check `ros2 topic list` and `ros2 topic hz`, that the robot's "
                "sensor stack is up, and that ROS_DOMAIN_ID matches. If the topic "
                "is publishing but nothing arrives here, it is a QoS mismatch -- "
                "see this class's docstring."
            )

        return replace(
            intrinsics_from_camera_info(self._info),
            depth_scale=depth_quantisation(self._depth_msg.encoding),
        )

    def _decode(self, msg) -> np.ndarray:
        return decode_depth_image(
            msg.data, msg.height, msg.width, msg.encoding, msg.step, msg.is_bigendian
        )

    def __iter__(self) -> Iterator[Frame]:
        while not self._closed:
            previous = self._depth_msg
            # Bind `previous` as a default rather than closing over it: the
            # lambda is consumed immediately so late binding is harmless today,
            # but the pattern is one edit away from being a real bug.
            if not self._spin_until(lambda prev=previous: self._depth_msg is not prev):
                return  # The stream stopped; ending the iteration says so.

            msg = self._depth_msg
            color = None
            if self._want_color and self._color_msg is not None:
                cm = self._color_msg
                if cm.encoding in ("rgb8", "bgr8"):
                    img = np.frombuffer(cm.data, dtype=np.uint8).reshape(
                        cm.height, cm.step // 3, 3
                    )[:, : cm.width]
                    color = img[..., ::-1].copy() if cm.encoding == "rgb8" else img.copy()

            stamp = msg.header.stamp
            yield Frame(
                depth_m=self._decode(msg),
                intrinsics=self.intrinsics,
                color=color,
                index=self._index,
                timestamp_s=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            )
            self._index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._node.destroy_node()
        if self._owns_context:
            with contextlib.suppress(Exception):
                self._rclpy.shutdown()

    def __enter__(self) -> X2RgbdSource:
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
            # Recordings made before this field was saved fall back to the D435
            # value, which is what they were captured with.
            subpixel_px=float(data["subpixel_px"]) if "subpixel_px" in data.files else 0.08,
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
        # subpixel_px is part of the noise model, not decoration. Omitting it
        # replays a Gemini 335 recording under a D435's disparity noise, which
        # under-states sigma by ~1.5x and shifts every threshold that scales
        # with range_sigma -- silently, since the depth itself is unchanged.
        "subpixel_px": intr.subpixel_px,
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
