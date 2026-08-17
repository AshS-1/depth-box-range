# boxrange

Measure the distance to a box from RealSense depth camera input.

Depth frame in, tracked metric distance out, ~30 FPS on a laptop CPU. No
training, no CAD model, no GPU.

![overlay](docs/overlay.png)

## What it does

Finds box-shaped objects sitting on a floor and reports how far away they are,
with an uncertainty and a confidence score. Per frame it fits the ground plane,
isolates what stands on it, fits an oriented box, and measures the distance —
then tracks boxes across frames so a static box doesn't jitter.

Because "distance to a box" is ambiguous, four numbers come back:

| field | meaning | use for |
|---|---|---|
| `surface_m` | Euclidean, camera to nearest point on the box | collision, stopping distance |
| `axial_m` | nearest box surface along +z | forward-driving robots |
| `centroid_m` | Euclidean to box centre | grasping, planner targets |
| `measured_m` | robust nearest observed depth, model-free | cross-check on the fit |

Plus `sigma_m` (1-sigma), `confidence` (0–1), `visible_faces`, and the fitted
box pose and extents.

## Run it on your computer

No camera needed — a synthetic source renders a box with a modelled stereo
noise profile.

```bash
uv pip install -e ".[dev]"
python -m boxrange --selftest
```

`--selftest` places a box at known distances, measures each, and checks the
answer against truth. Exit code 0 on pass, so it works in CI.

```bash
python -m boxrange --frames 30 --distance 2.0
python -m boxrange --frames 0 --distance 2.0 --show
python -m boxrange --frames 5 --distance 2.5 --json
```

`--distance` sets the box *centre*; the pipeline reports distance to the
nearest *face*, so the two differ. The CLI prints the true face distance for
comparison. Also try `--yaw`, `--box-size`, `--no-noise`.

## Run it with a RealSense

```bash
pip install -e ".[realsense]"
python -m boxrange --source live --show
```

Record on the machine with the camera, analyse anywhere:

```bash
python -m boxrange --source live --record run.npz --frames 200
python -m boxrange --source npz --path run.npz --json
python -m boxrange --source bag --path scene.bag --json > ranges.jsonl
```

```python
from boxrange import BoxRangePipeline, RealSenseSource

pipeline = BoxRangePipeline()
with RealSenseSource(color=False) as source:
    for frame in source:
        for det in pipeline.process(frame):
            print(det.range.surface_m, det.range.sigma_m, det.range.confidence)
```

**Use `color=False` for pure ranging.** Enabling colour aligns depth into the
colour frame, which crops depth FOV from roughly 87°×58° to 69°×42° and
resamples depth at exactly the discontinuities segmentation depends on. Colour
is never used for measurement — only for the overlay background and as input to
an optional learned detector.

`pyrealsense2` has no wheel for macOS arm64, so it's an optional extra. The
synthetic and `.npz` paths run anywhere.

## Why it works

**Depth measures range directly.** A box is a cuboid primitive and the sensor
already observes distance, so geometry beats regressing it from RGB — no
training data, no CAD model, no labels.

**The box fit is plane-constrained.** Free 3D PCA is the obvious approach and
it's wrong: a depth camera sees only one or two faces, so the point mass is a
hollow L-shell and the principal axes tilt off the true edges. Locking one axis
to the ground normal and recovering the other two from the footprint's
minimum-area rectangle gets the true edge directions from a single visible
corner.

**Every threshold scales with the sensor noise model, not absolute
millimetres.** Stereo error grows as z², so a 5.7 cm residual is terrible at
1 m and textbook-perfect at 3.5 m where the noise floor *is* 5 cm. Clustering
tolerance, face tolerance, and confidence all derive from `range_sigma(z)`.
An earlier absolute-threshold version scored distance as if it were error and
silently dropped every detection past 2.5 m.

**Uncertainty is reported, not hidden.** Always read `sigma_m` — a box at 4 m
carries ~16× the variance of one at 1 m.

## Accuracy and limits

Against synthetic scenes with exact ground truth (0.40 × 0.30 × 0.25 m box,
D435-like intrinsics): **−7 mm at 1 m, −27 mm at 2 m, −107 mm at 3.8 m**, with
extents within ~2 cm out to 2.5 m. Run `--selftest` for the full table.

Two limits worth knowing:

- **The error is a bias, not noise.** It reads systematically *near* of truth,
  growing with range, because depth noise inflates the fitted footprint outward
  and pushes the near corner toward the camera. Reading near is the safe
  direction for obstacle avoidance. Past ~3 m the bias **exceeds the reported
  sigma**, so don't treat `interval_95` as covering truth at long range.
- **These are synthetic-model numbers.** Real D400 depth also carries
  fixed-pattern and thermal error this model omits. Validate against a real
  camera and a tape measure before trusting them.

Needs a visible support plane, and resolves a cuboid's pose only up to a
cuboid's symmetry — it can't tell you which way the label faces. For full 6D
pose of a specific object, heavy occlusion, or telling identical boxes apart,
plug a learned detector into the `Detector` protocol in `boxrange/segment.py`;
the network localises, depth still measures.

## Tests

```bash
pytest -q
ruff check boxrange/ tests/
```

48 tests. `test_accuracy.py` asserts metric results against known truth;
`test_robustness.py` covers degenerate input, one case per bug found by
fuzzing. RuntimeWarnings are errors.

Coordinate convention throughout is the depth optical frame: +x right, +y down,
+z forward (OpenCV / RealSense).
