"""Command line entry point: ``python -m boxrange``."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .frames import (
    NpzSource,
    RealSenseSource,
    SyntheticScene,
    SyntheticSource,
    record_npz,
)
from .pipeline import BoxRangePipeline, detection_to_dict
from .segment import PlaneClusterDetector


def build_source(args):
    if args.source == "live":
        return RealSenseSource(width=args.width, height=args.height, fps=args.fps)
    if args.source == "bag":
        if not args.path:
            raise SystemExit("--path is required for --source bag")
        return RealSenseSource(bag_path=args.path)
    if args.source == "npz":
        if not args.path:
            raise SystemExit("--path is required for --source npz")
        return NpzSource(args.path)
    scene = SyntheticScene(
        forward_m=args.distance,
        yaw=np.deg2rad(args.yaw),
        box_size=tuple(args.box_size),
    )
    # frames=0 means "keep going" everywhere else, so don't hand the synthetic
    # source a zero-length stream.
    return SyntheticSource(scene, frames=args.frames or 300, noise=not args.no_noise)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="boxrange",
        description="Measure the distance to a box from RealSense depth input.",
    )
    p.add_argument(
        "--source", default="synthetic", choices=("live", "bag", "npz", "synthetic"),
        help="where frames come from (default: synthetic, needs no hardware)",
    )
    p.add_argument("--path", help="file for --source bag / npz")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
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
    p.add_argument("--record", help="save frames to this .npz and exit")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--selftest", action="store_true",
        help="measure known-truth scenes and print a pass/fail table (no camera needed)",
    )

    args = p.parse_args(argv)

    if args.selftest:
        from .selftest import run

        return 0 if run() else 1

    source = build_source(args)

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

    pipeline = BoxRangePipeline(
        detector=PlaneClusterDetector(),
        max_boxes=args.max_boxes,
        min_confidence=args.min_confidence,
    )

    saved = False
    latencies: list[float] = []
    seen = 0

    try:
        for frame in source:
            t0 = time.perf_counter()
            detections = pipeline.process(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            seen += 1

            for det in detections:
                if args.json:
                    print(json.dumps(detection_to_dict(det, frame)), flush=True)
                elif not args.quiet:
                    print(f"[{frame.index:4d}] track {det.track_id}  {det.range}")

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
