# Skylight-polarization intrinsic calibration demo

This package runs the relative-rotation and time-separated static calibration examples from raw polarization images. It contains executable source, input data and this README. Outputs are generated on demand.

## Quick start

Tested with Python 3.14.3. Open a terminal in this folder. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -B run_demo.py
```

On Linux/macOS, activate a Python environment, install `requirements.txt`, then run:

```sh
OPENBLAS_NUM_THREADS=1 python -B run_demo.py
```

Run from the supplied raw data without the original research project, MATLAB installation or external INS files. The first run creates `results/`. Use `--output-dir PATH` to save elsewhere. For subsequent commands, use the Python executable from the environment where the dependencies were installed. Once a default run has completed, `python -B run_demo.py --plots-only` regenerates its figures after checking the saved input/source hashes.

## Files

| File or folder | Purpose |
| --- | --- |
| `run_demo.py` | Run both calibrations and generate the comparison plot |
| `calibration_rotation.py` | Relative-rotation calibration |
| `calibration_static_heading_domain.py` | Time-separated static calibration |
| `demo_common.py` | Raw image loading, Stokes/AoP, E-vectors, solar/heading calculation and output |
| `aop_compensation.py` | Fixed empirical AoP preprocessing |
| `principal_direction_prior.py`, `guarded_direction_fit.py` | Direction-prior construction and optimizer |
| `focal_regularization.py`, `residual_weighting.py` | Explicit objective settings; default residual weights are equal |
| `example_outputs.py` | Output checks, summary table and six-panel plot |
| `requirements.txt` | Pinned Python dependencies |
| `data/rotation/` | 14 raw images, configuration and per-frame reference/time manifest |
| `data/static/` | 26 raw images, configuration and per-frame reference/time manifest |
| `data/skybd25.mat` | Sensor gain/bias coefficients read by SciPy; required for preprocessing |
| `data/provenance.json` | Source and selection information |

The JSON configurations and CSV manifests are required inputs, not temporary files. Intrinsics use original-image pixels: `[f, u0(row), v0(column)]`.

## Method and data

Both modes initialize at `[1300, 360, 640]`. The same vector supplies the direction-prior residuals, reconstructed solar directions and line origin; the inferred axis is fixed during fitting. There is no separate prior-reference intrinsic vector. The focal initial value of 1300 px was supplied by the camera according to the author and is not treated as known ground truth.

The rotation example retains one raw frame per position from the April 19, 2026 acquisition: 14 positions including the start/end of one revolution, with 13 adjacent intervals. The static example retains 26 frames from the April 25, 2026 acquisition spanning about 102 minutes. Both contain original 720 × 1280 16-bit mosaics. The static subset comes from the manuscript-associated 52-frame acquisition; it is not a separate independent collection. These selected subsets demonstrate the computation and are not an exact numerical reproduction of Table 1 or Figure 9. Selection and earlier tuning were exploratory.

Both examples use matched INS headings stored in `frames.csv`. Rotation uses adjacent INS heading differences, while static uses differences relative to a fixed reference frame. The historical manuscript rotation experiment describes mechanical turntable references; this demo uses the documented INS reference instead. Reference values are fixed before fitting. The rotation increments are not adjusted to force their sum to 360°. Manifest timestamps use the inherited camera/INS matching procedure; timing accuracy has not been independently established here.

### Objective and focal stabilization

With all angular differences wrapped to `[-90°, 90°)`, the data residuals are:

```text
rotation: e_j = wrap180((psi[j+1]-psi[j]) - (INS[j+1]-INS[j]))
static:   e_t = wrap180((psi[t]-psi[r]) - (INS[t]-INS[r]))

default J = mean(e**2) + direction_penalty
            + 0.02 * ((f - 1300) / 130)**2
```

The final quadratic term is an explicit regularized extension of the manuscript objectives. Its weight and scale are engineering settings, not measured uncertainty. The direction penalty is a weak, normalized perpendicular-distance penalty to the line through the initial principal point. Bounds, prior settings, compensation coefficients and optimizer seed are in each `config.json`. Default weighting is equal and no fitted constant evaluation bias is subtracted (`fixed_bias_deg=0`). The same declared +45° coordinate convention applies before and after calibration.

Under the level-camera, heading-based observations used here, focal length is weakly identified. The penalty stabilizes its value near the supplied initial value; this does not establish independent recovery of physical focal length. Both calibration modes use this regularized objective as part of the supplied pipeline.

### Polarization preprocessing

The input images undergo sensor gain/bias correction and measured-analyzer inversion to obtain Stokes Q/U and AoP. The empirical AoP compensation originates from the authors’ subsequent work and is outside the methodological contributions of the present manuscript. It is a fixed preprocessing stage in this demo, using the same settings before and after intrinsic calibration. The reported results describe this combined pipeline. For amplitude `A=hypot(Q,U)` and `theta=atan2(U,Q)/2`:

```text
delta = (dU*cos(2*theta) - dQ*sin(2*theta)) / (2*A)
theta_corrected = theta - clip(delta, -max_delta, +max_delta)
```

Pixels below the configured minimum amplitude receive zero correction. Profiles are selected from the current ROI's `hypot(mean(Q),mean(U))`:

| ROI amplitude | dQ | dU | Pixel minimum A | Maximum correction |
| --- | ---: | ---: | ---: | ---: |
| < 1600 | 120 | -125 | 150 | 18° |
| 1600–10000 | 57.84 | -132.24 | 300 | 12° |
| > 10000 | 28 | -65 | 1000 | 2° |

Coefficients are fixed in the supplied configurations and inherited from historical heading-error analysis; they are not fitted by this demo. Their development-data overlap is unverified, so the results are not blind validation of compensation. Basic Stokes-to-AoP reconstruction remains enabled. This implementation does not apply geometric undistortion.

### Angle and coordinate conventions in this implementation

Both supplied configurations select `model_id=1`. The variables used by `demo_common.py` are:

```text
x = column - v0; y = row - u0
alpha = atan2(x, y)
gamma = atan(hypot(x, y) / f)
phi = atan(cos(gamma) * tan(alpha - (AoP + pi/2)))

e_sensor = [sin(alpha)*cos(gamma)*cos(phi) - cos(alpha)*sin(phi),
            cos(alpha)*cos(gamma)*cos(phi) + sin(alpha)*sin(phi),
           -sin(gamma)*cos(phi)]
e_level = Cpl @ e_sensor
psi_raw_deg = degrees(SunAz - atan2(s_level[0], s_level[1]))
```

Here `s_level` is the smallest-eigenvalue direction of the E-vector scatter matrix. The `pi/2` term is part of the implemented AoP-to-vector convention; it is distinct from the 180-degree ambiguity of a linear-polarization axis. With both heading and INS expressed in degrees, the per-frame evaluation uses `wrap180(psi_raw_deg + 45 - INS_deg)`, with zero fixed-bias subtraction. The 45-degree offset affects absolute heading evaluation and cancels in the relative-angle fitting residuals. Neither angle offset is adjusted by the intrinsic optimizer. These explicit implementation conventions must be distinguished when comparing formulas expressed in other axes.

## Outputs and expected default results

The default run generates per-mode `*_summary.json` and `*_frames.csv`. `results/example_results/` contains the parameter/error summary, pairwise consistency residuals and `example_calibration.png` (also PDF/SVG). No precomputed results are required to run this package.

| Method | Initial heading RMSE | Calibrated heading RMSE | Relative-residual RMS before → after |
| --- | ---: | ---: | ---: |
| Rotation | 0.518086° | 0.233610° | 0.314185° → 0.215241° |
| Static | 1.540769° | 0.245645° | 0.698251° → 0.043675° |

Expected default intrinsic vectors are approximately `[1300.169290, 384.693042, 637.001152]` and `[1299.974730, 392.282990, 652.335441]`, respectively. Floating-point/optimizer differences can affect final digits across environments.

The plot's left column shows the relative-angle residuals used by the data term. The middle column shows reconstructed heading errors against the bundled INS reference. The right column summarizes those heading-error distributions. They are calibration-set diagnostics, not pixel-level AoP errors or independent validation. The same preprocessing is used on both sides of each intrinsic comparison.
