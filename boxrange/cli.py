"""Command line entry point: ``python -m boxrange``."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .frames import (
    X2_DEPTH_IMAGE,
    NpzSource,
    OrbbecSource,
    RealSenseSource,
    SyntheticScene,
    SyntheticSource,
    X2RgbdSource,
    record_npz,
)
from .intrinsics import CAMERA_PRESETS, PRESET_RESOLUTIONS
from .pipeline import BoxRangePipeline, detection_to_dict
from .segment import PlaneClusterDetector


def build_source(args):
    if args.source == "x2":
        # No width/height: the robot publishes one depth stream and its
        # CameraInfo is authoritative, so there is nothing to negotiate.
        return X2RgbdSource(color=not args.no_color, reliable=args.ros_reliable)
    if args.source == "orbbec":
        # 0 means "whatever the device offers by default" to the Orbbec SDK.
        return OrbbecSource(
            width=args.width or 0, height=args.height or 0, fps=args.fps,
            color=not args.no_color,
        )
    if args.source == "live":
        return RealSenseSource(
            width=args.width or 640, height=args.height or 480, fps=args.fps,
            color=not args.no_color,
        )
    if args.source == "bag":
        if not args.path:
            raise SystemExit("--path is required for --source bag")
        return RealSenseSource(bag_path=args.path, color=not args.no_color)
    if args.source == "npz":
        if not args.path:
            raise SystemExit("--path is required for --source npz")
        return NpzSource(args.path)
    scene = SyntheticScene(
        forward_m=args.distance,
        yaw=np.deg2rad(args.yaw),
        box_size=tuple(args.box_size),
    )
    # Simulating a specific camera matters: the X2's Gemini 335 has a 90 deg FOV
    # and roughly 3x the range noise of a D435 at the same distance, so a result
    # measured under D435 intrinsics does not transfer to the robot.
    default_w, default_h = PRESET_RESOLUTIONS[args.camera]
    intrinsics = CAMERA_PRESETS[args.camera](
        args.width or default_w, args.height or default_h
    )
    # frames=0 means "keep going" everywhere else, so don't hand the synthetic
    # source a zero-length stream.
    return SyntheticSource(
        scene, intrinsics, frames=args.frames or 300, noise=not args.no_noise
    )


def _report_stream(intr, hint: str) -> int:
    """Print what a source actually negotiated. Shared by both probes."""
    print(f"depth stream {intr.width}x{intr.height}  "
          f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"depth_scale={intr.depth_scale:.6f} m/unit  baseline={intr.baseline_m:.4f} m")
    print(f"horizontal FOV {np.rad2deg(2 * np.arctan(intr.width / 2 / intr.fx)):.0f} deg")
    print(f"modelled noise: +/-{intr.range_sigma(1.0) * 1000:.0f} mm at 1 m, "
          f"+/-{intr.range_sigma(3.0) * 1000:.0f} mm at 3 m")
    print(f"\nOK -- camera is streaming. Run: {hint}")
    return 0


class _FoundationPoseEngine:
    """Adapts :class:`DepthOnlyFoundationPose` to the pipeline's process() shape.

    One pose per frame becomes a zero- or one-element detection list, so the
    JSON writer, the overlay and the timing code below are shared verbatim
    between the two engines and their outputs stay comparable.
    """

    def __init__(self, estimator) -> None:
        self._estimator = estimator

    def process(self, frame):
        result = self._estimator.process(frame)
        return [] if result is None else [result.to_detection()]


def build_engine(args):
    if args.engine == "geometric":
        return BoxRangePipeline(
            detector=PlaneClusterDetector(),
            max_boxes=args.max_boxes,
            min_confidence=args.min_confidence,
        )

    from .foundationpose import (
        DepthOnlyFoundationPose,
        GeometricEstimator,
        require_foundationpose,
    )

    stub = args.engine == "foundationpose-stub"
    if not stub:
        # Fail now rather than on the first frame. The estimator is built lazily
        # because it needs the box extents, which may only be known once a frame
        # has arrived -- but on a robot that means opening the camera, streaming,
        # and only then reporting that the environment was never going to work.
        require_foundationpose()

    if stub:
        print(
            "WARNING: --engine foundationpose-stub does not run FoundationPose. "
            "It swaps in a geometric solver to exercise the mask, mesh, "
            "pseudo-colour and pose-to-range plumbing without a GPU. Do not "
            "report its numbers as FoundationPose results.",
            file=sys.stderr,
        )
    return _FoundationPoseEngine(
        DepthOnlyFoundationPose(
            extents_m=args.box_extents,
            estimator=GeometricEstimator() if stub else None,
        )
    )


def probe_orbbec() -> int:
    """Report connected Orbbec devices, then actually open the depth stream.

    For a Gemini 335 plugged in over USB. On the X2 itself the head camera is
    reached over ROS 2 instead -- see :func:`probe_x2`.

    Opening the stream is the only real test: enumeration succeeding tells you
    the USB link is up, not that the mode you asked for exists or that another
    process has not already claimed the device.
    """
    try:
        import pyorbbecsdk as ob
    except ImportError:
        print("pyorbbecsdk not installed. `pip install -e \".[orbbec]\"` "
              "(Linux/Windows wheels; macOS needs a source build).", file=sys.stderr)
        return 1

    devices = ob.Context().query_devices()
    if devices.get_count() == 0:
        print("no Orbbec device found. Check the USB 3 Type-C cable, and on Linux "
              "that the SDK's udev rules are installed "
              "(99-obsensor-libusb.rules, then udevadm control --reload).",
              file=sys.stderr)
        return 1

    for i in range(devices.get_count()):
        info = devices.get_device_by_index(i).get_device_info()
        print(f"{info.get_name()}  serial {info.get_serial_number()}  "
              f"firmware {info.get_firmware_version()}")

    try:
        source = OrbbecSource(color=False, warmup_frames=2)
    except Exception as exc:
        print(f"device found but streaming failed: {exc}", file=sys.stderr)
        return 1

    intr = source.intrinsics
    source.close()
    return _report_stream(intr, "python -m boxrange --source orbbec --no-color --show")


def probe_x2() -> int:
    """Report whether the X2's head RGB-D topics are actually publishing.

    Distinguishes the failures that all look like a dead pipeline: no ROS 2 at
    all, ROS 2 up but the sensor stack down, the topic advertised but silent, and
    the topic publishing to a subscriber that cannot receive it because the QoS
    is incompatible.
    """
    try:
        import rclpy
    except ImportError:
        print("rclpy not found. Source the robot's ROS 2 setup.bash "
              "(ROS 2 does not install from pip).", file=sys.stderr)
        return 1

    owns = not rclpy.ok()
    if owns:
        rclpy.init()
    node = rclpy.create_node("boxrange_probe")
    try:
        topics = dict(node.get_topic_names_and_types())
        found = [t for t in topics if "rgbd_head_front" in t]
        if not found:
            print("no /aima/hal/sensor/rgbd_head_front/* topics advertised. Is the "
                  "robot's sensor stack running, and does ROS_DOMAIN_ID match?",
                  file=sys.stderr)
            return 1
        for topic in sorted(found):
            print(f"{topic}  {','.join(topics[topic])}")
        if X2_DEPTH_IMAGE not in topics:
            print(f"\nwarning: {X2_DEPTH_IMAGE} is not among them; depth is what "
                  "this package needs.", file=sys.stderr)
    finally:
        node.destroy_node()
        if owns:
            rclpy.shutdown()

    # Advertised is not the same as receivable, so actually subscribe.
    try:
        source = X2RgbdSource(color=False, timeout_s=10.0)
    except Exception as exc:
        print(f"topics advertised but no frames received: {exc}", file=sys.stderr)
        return 1

    intr = source.intrinsics
    source.close()
    return _report_stream(intr, "python -m boxrange --source x2 --show")


def probe_realsense() -> int:
    """Report what the RealSense stack can actually see, before streaming.

    Separates the three failures that all look identical from a dead pipeline:
    the library is missing, no device is plugged in, or the device is there but
    the stream config was refused (usually USB 2).
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed. `pip install -e \".[realsense]\"`",
              file=sys.stderr)
        return 1

    devices = list(rs.context().query_devices())
    if not devices:
        print("no RealSense device found. Check the cable (USB 3, blue port), "
              "and on Linux that udev rules are installed.", file=sys.stderr)
        return 1

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        usb = "unknown"
        if dev.supports(rs.camera_info.usb_type_descriptor):
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"{name}  serial {serial}  USB {usb}")
        if usb.startswith("2"):
            print("  warning: USB 2 link. Depth resolution and frame rate will "
                  "be limited; use a USB 3 port and cable.")

    # Actually opening the stream is the only real test of the config.
    try:
        source = RealSenseSource(color=False, warmup_frames=2)
    except Exception as exc:
        print(f"device found but streaming failed: {exc}", file=sys.stderr)
        return 1

    intr = source.intrinsics
    source.close()
    return _report_stream(intr, "python -m boxrange --source live --no-color --show")


def probe(source: str) -> int:
    """Probe whichever camera ``--source`` names.

    With no camera named there is no way to know which SDK the user cares about,
    so try both and report each. Returning 0 if *either* works keeps the exit
    code meaning "a usable depth camera is attached".
    """
    if source == "x2":
        return probe_x2()
    if source == "orbbec":
        return probe_orbbec()
    if source in ("live", "bag"):
        return probe_realsense()

    # Flush: these headers go to stdout while the failure messages below go
    # to stderr, and without flushing the two arrive out of order.
    print("== AgiBot X2 head RGB-D (ROS 2) ==", flush=True)
    x2 = probe_x2()
    print("\n== Orbbec (Gemini 335 over USB) ==", flush=True)
    orbbec = probe_orbbec()
    print("\n== RealSense ==", flush=True)
    realsense = probe_realsense()
    return 0 if 0 in (x2, orbbec, realsense) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="boxrange",
        description="Measure the distance to a box from depth-camera input.",
    )
    p.add_argument(
        "--source", default="synthetic",
        choices=("x2", "orbbec", "live", "bag", "npz", "synthetic"),
        help="where frames come from (default: synthetic, needs no hardware). "
             "x2 = the AgiBot X2's head RGB-D camera over its ROS 2 "
             "interface; orbbec = a Gemini 335 straight over USB; "
             "live/bag = RealSense",
    )
    p.add_argument(
        "--camera", default="d435", choices=tuple(CAMERA_PRESETS),
        help="synthetic/selftest: which sensor to model. gemini335 is the X2's "
             "head camera and carries ~3x a D435's range noise, so its accuracy "
             "table is a different table (default: d435)",
    )
    p.add_argument("--path", help="file for --source bag / npz")
    # Default None, not a number: the right resolution depends on which camera
    # is in play, and a hardcoded 640x480 would silently model the X2's 8:5
    # sensor at 4:3 -- a 74 degree vertical FOV it does not have.
    p.add_argument("--width", type=int, help="depth width (default: per --camera / device)")
    p.add_argument("--height", type=int, help="depth height (default: per --camera / device)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--ros-reliable", action="store_true",
        help="x2: subscribe RELIABLE instead of BEST_EFFORT. The default is "
             "BEST_EFFORT because it is compatible with publishers of either "
             "kind, while a RELIABLE subscriber against a BEST_EFFORT publisher "
             "silently receives nothing",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="live/bag: depth only. Recommended for ranging -- aligning depth to "
             "the colour frame crops depth FOV (~87x58 deg down to ~69x42) and "
             "resamples depth at the discontinuities segmentation relies on",
    )
    p.add_argument("--frames", type=int, default=30, help="frames to process (0 = all)")

    p.add_argument("--distance", type=float, default=1.6, help="synthetic: box distance")
    p.add_argument("--yaw", type=float, default=25.0, help="synthetic: box yaw, degrees")
    p.add_argument("--box-size", type=float, nargs=3, default=[0.40, 0.30, 0.25],
                   metavar=("D", "W", "H"), help="synthetic: true box size, metres")
    p.add_argument("--no-noise", action="store_true", help="synthetic: disable sensor noise")

    p.add_argument("--max-boxes", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.05)
    p.add_argument("--json", action="store_true", help="one JSON record per detection")
    p.add_argument("--show", action="store_true", help="live overlay window")
    p.add_argument("--save-overlay", help="write the first overlay frame to this path")
    p.add_argument(
        "--save-pseudo-rgb",
        help="write the shading image that stands in for the colour stream to "
             "this path, then carry on. Worth looking at before trusting any "
             "FoundationPose result: if it is flat grey, the network is getting "
             "no appearance signal and only the depth half of its input is real",
    )
    p.add_argument("--record", help="save frames to this .npz and exit")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--selftest", action="store_true",
        help="measure known-truth scenes and print a pass/fail table (no camera needed)",
    )

    p.add_argument(
        "--probe", action="store_true",
        help="report connected cameras and stream settings, then exit. Probes "
             "the SDK matching --source, or both if none is named",
    )

    p.add_argument(
        "--engine", default="geometric",
        choices=("geometric", "foundationpose", "foundationpose-stub"),
        help="geometric = the plane-constrained depth fit (default; no GPU). "
             "foundationpose = 6D pose from NVlabs FoundationPose driven by "
             "depth alone, which needs CUDA, nvdiffrast and the pretrained "
             "weights. foundationpose-stub swaps the network for a geometric "
             "solver to check the plumbing without a GPU -- its numbers are NOT "
             "FoundationPose's",
    )
    p.add_argument(
        "--box-extents", type=float, nargs=3, metavar=("D", "W", "H"),
        help="foundationpose: true box size in metres, used to build the mesh. "
             "Measure it if you can -- omitted, it is taken from the depth fit "
             "on the first frame, and a wrong mesh biases every pose after it",
    )

    args = p.parse_args(argv)

    if args.probe:
        return probe(args.source)

    if args.selftest:
        from .selftest import run

        return 0 if run(camera=args.camera) else 1

    try:
        source = build_source(args)
    except ImportError as exc:
        # A camera SDK that is not installed on this machine is an expected
        # outcome, not a bug -- one line, not a traceback. Same treatment as a
        # missing FoundationPose environment below.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # On synthetic input we know the answer, so print it. --distance places the
    # box *centre* at that range along the floor, while the pipeline reports the
    # distance to the nearest *face* -- without this the two look like a 15 cm
    # error when they are measuring different things.
    if args.source == "synthetic" and not args.json and not args.quiet:
        from .ranging import closest_point_on_box

        truth = float(np.linalg.norm(closest_point_on_box(source.truth_box())))
        print(
            f"ground truth: nearest face at {truth:.3f} m "
            f"(--distance {args.distance} sets the box centre on the floor)",
            file=sys.stderr,
        )

    if args.record:
        n = record_npz(source, args.record, max_frames=args.frames or 150)
        print(f"recorded {n} frames -> {args.record}", file=sys.stderr)
        source.close()
        return 0

    try:
        engine = build_engine(args)
    except ImportError as exc:
        # A missing CUDA/FoundationPose environment is an expected outcome on a
        # laptop, not a bug. Report it as one line instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        source.close()
        return 1

    saved = False
    saved_rgb = False
    latencies: list[float] = []
    seen = 0

    try:
        for frame in source:
            t0 = time.perf_counter()
            detections = engine.process(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            seen += 1

            for det in detections:
                if args.json:
                    record = detection_to_dict(det, frame)
                    # Name the engine in the record. Two engines writing the same
                    # keys into the same log with no way to tell them apart is a
                    # good way to publish a stub's numbers as FoundationPose's.
                    record["engine"] = args.engine
                    print(json.dumps(record), flush=True)
                elif not args.quiet:
                    print(f"[{frame.index:4d}] track {det.track_id}  {det.range}")

            if args.save_pseudo_rgb and not saved_rgb:
                import cv2

                from .foundationpose import depth_to_pseudo_rgb

                cv2.imwrite(
                    args.save_pseudo_rgb,
                    depth_to_pseudo_rgb(frame.depth_m, frame.intrinsics),
                )
                saved_rgb = True

            if (args.show or args.save_overlay) and not (saved and not args.show):
                from .viz import render_frame

                overlay = render_frame(
                    frame.depth_m, detections, frame.intrinsics, background=frame.color
                )
                if args.save_overlay and not saved:
                    import cv2

                    cv2.imwrite(args.save_overlay, overlay)
                    saved = True
                if args.show:
                    import cv2

                    cv2.imshow("boxrange", overlay)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

            if args.frames and seen >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    if latencies and not args.quiet and not args.json:
        arr = np.asarray(latencies)
        print(
            f"\n{seen} frames | {arr.mean():.1f} ms/frame mean, "
            f"{np.percentile(arr, 95):.1f} ms p95 | {1000.0 / max(arr.mean(), 1e-6):.1f} FPS",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
