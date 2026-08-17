"""Does it actually work? Run the pipeline against scenes with known truth.

Distinct from the test suite: this prints the numbers so you can *see* the
measurement tracking truth, rather than just getting a green bar. Answers the
only question that matters -- put a box at a known distance, does it report
that distance?
"""

from __future__ import annotations

import time

import numpy as np

from .frames import SyntheticScene, SyntheticSource
from .intrinsics import CameraIntrinsics
from .pipeline import BoxRangePipeline
from .ranging import closest_point_on_box

# D435-like depth stream.
_INTR = CameraIntrinsics(width=640, height=480, fx=385.0, fy=385.0, cx=320.0, cy=240.0)


def run(distances=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), seeds: int = 3) -> bool:
    box = (0.40, 0.30, 0.25)
    print(f"Box {box[0]:.2f} x {box[1]:.2f} x {box[2]:.2f} m on a floor, "
          f"D435-like depth, {seeds} noisy renders per distance.\n")
    print(f"{'truth':>8} {'reported':>10} {'error':>9} {'+/-':>8} "
          f"{'size err':>9} {'faces':>6} {'conf':>6}   verdict")
    print("-" * 74)

    all_ok = True
    latencies: list[float] = []

    for dist in distances:
        scene = SyntheticScene(forward_m=dist, box_size=box)
        truth = float(np.linalg.norm(closest_point_on_box(scene.truth_box())))

        reported, sizes, sigmas, faces, confs = [], [], [], [], []
        for seed in range(seeds):
            pipeline = BoxRangePipeline()
            for frame in SyntheticSource(scene, intrinsics=_INTR, frames=1, seed=seed):
                t0 = time.perf_counter()
                dets = pipeline.process(frame)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                if dets:
                    r = dets[0].range
                    reported.append(r.surface_m)
                    sizes.append(abs(dets[0].box.extents[0] - box[0]))
                    sigmas.append(r.sigma_m)
                    faces.append(r.visible_faces)
                    confs.append(r.confidence)

        if not reported:
            print(f"{truth:7.3f}m {'MISSED':>10} {'':>9} {'':>8} {'':>9} {'':>6} {'':>6}   FAIL")
            all_ok = False
            continue

        got = float(np.mean(reported))
        err = got - truth
        # Pass if within the sensor's own noise floor, scaled -- demanding 1 cm
        # at 4 m would be demanding better than the hardware can do.
        tol = max(0.05, 3.0 * float(_INTR.range_sigma(truth)))
        ok = abs(err) < tol
        all_ok &= ok
        print(f"{truth:7.3f}m {got:9.3f}m {err:+8.3f}m {np.mean(sigmas):7.3f}m "
              f"{np.mean(sizes):8.3f}m {np.mean(faces):6.1f} {np.mean(confs):6.2f}   "
              f"{'ok' if ok else 'FAIL'}")

    arr = np.asarray(latencies)
    print("-" * 74)
    print(f"{arr.mean():.1f} ms/frame ({1000.0 / max(arr.mean(), 1e-9):.0f} FPS) on this CPU")
    print("\nNote: error is negative by design -- the fit reads slightly NEAR of")
    print("truth, the safe direction for obstacle avoidance. Past ~3 m that bias")
    print("exceeds the reported sigma; see README.")
    print("\nPASS -- ranging tracks ground truth." if all_ok else "\nFAIL -- see rows above.")
    return all_ok
