"""Shared I/O and E-vector reconstruction for the self-contained reviewer demo.

u is the original image row; v is the column. Intrinsics use original pixels.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
from pysolar import solar
from aop_compensation import correct_roi_aop

ROOT = Path(__file__).resolve().parent


def read_config(mode, config_path=None):
    path = Path(config_path).resolve() if config_path else ROOT / "data" / mode / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["mode"] != mode:
        raise ValueError(f"Expected {mode} configuration: {path}")
    if "nominal_intrinsics_px" in cfg:
        raise ValueError("Obsolete separate direction-prior input; remove nominal_intrinsics_px and use initial_intrinsics_px")
    cfg["config_path"] = path
    cfg["manifest_path"] = (path.parent / cfg["manifest"]).resolve()
    cfg["calibration_path"] = (path.parent / cfg["sensor_calibration"]).resolve()
    return cfg


def read_manifest(cfg):
    path = cfg["manifest_path"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        timestamp = datetime.fromisoformat(row["timestamp_utc"])
        if timestamp.tzinfo is None:
            raise ValueError("Manifest timestamps must include a UTC offset")
        record = dict(row)
        record["timestamp_utc"] = timestamp.astimezone(timezone.utc)
        record["reference_heading_deg"] = float(row["reference_heading_deg"])
        record["image_path"] = (path.parent / row["image"]).resolve()
        if not record["image_path"].is_file():
            raise FileNotFoundError(record["image_path"])
        records.append(record)
    if not records:
        raise ValueError("Empty frame manifest")
    return records


def sensor_to_level(cfg):
    roll, pitch = np.radians([cfg["roll_deg"], cfg["pitch_deg"]])
    cr = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)],
                   [0, -np.sin(roll), np.cos(roll)]])
    cp = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0],
                   [np.sin(pitch), 0, np.cos(pitch)]])
    return (cr @ cp).T @ np.asarray(cfg["sensor_to_platform"], dtype=float)


def load_dataset(cfg, limit=None):
    records = read_manifest(cfg)
    if limit is not None:
        records = records[:limit]
    calibration = loadmat(cfg["calibration_path"])
    gain = calibration["k"].ravel()
    bias = calibration["b"].ravel() * cfg["kb_bias_scale"]
    alpha = np.asarray(cfg["analyzer_angles_rad"])
    design = np.column_stack((np.cos(2*alpha), np.sin(2*alpha), np.ones(4)))
    inverse = np.linalg.pinv(design)
    transform = sensor_to_level(cfg)
    initial = np.asarray(cfg["initial_intrinsics_px"])
    bounds = np.asarray(cfg["bounds_px"])
    radius = float(cfg["mask_radius_px"])
    data, shape = [], None
    for record in records:
        image_bytes = record["image_path"].read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != record["sha256"]:
            raise ValueError(f"Image checksum mismatch: {record['image']}")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or any(n % 2 for n in image.shape):
            raise ValueError(f"Expected even-sized grayscale polarizer mosaic: {record['image']}")
        if shape is not None and image.shape != shape:
            raise ValueError("Example frames must have identical dimensions")
        shape = image.shape
        values = image.astype(float)
        channels = np.stack((values[0::2, 0::2], values[0::2, 1::2],
                             values[1::2, 0::2], values[1::2, 1::2]))
        channels = gain[:, None, None]*channels + bias[:, None, None]
        stokes = np.tensordot(inverse, channels, axes=(1, 0))
        aop = 0.5*np.arctan2(stokes[1], stokes[0])
        yi, xi = np.where(np.isfinite(aop))
        step = int(cfg["sample_step"])
        yi, xi = yi[::step], xi[::step]
        rows, cols = 2*yi+1, 2*xi
        # Exclude only pixels outside every admissible circular ROI.
        union = ((rows >= min(bounds[1, 0],initial[1])-radius) & (rows <= max(bounds[1, 1],initial[1])+radius)
                 & (cols >= min(bounds[2, 0],initial[2])-radius) & (cols <= max(bounds[2, 1],initial[2])+radius))
        yi, xi, rows, cols = yi[union], xi[union], rows[union], cols[union]
        if not len(rows):
            raise ValueError("No finite AoP samples in the admissible search region")
        # Original reference-frame score, including its half-resolution centre.
        q_dolp, u_dolp = channels[0]-channels[3], channels[2]-channels[1]
        intensity = np.sum(channels, axis=0)/2
        dolp = np.hypot(q_dolp, u_dolp)/(intensity+1e-6)
        yy, xx = np.ogrid[:dolp.shape[0], :dolp.shape[1]]
        score_mask = ((yy-initial[1]/2)**2+(xx-initial[2]/2)**2 <= (radius/2)**2)
        score = float(np.nanmean(dolp[score_mask]))
        timestamp = record["timestamp_utc"]
        latitude, longitude = cfg["location"]["latitude"], cfg["location"]["longitude"]
        azimuth = solar.get_azimuth(latitude, longitude, timestamp)
        zenith = 90-solar.get_altitude(latitude, longitude, timestamp)
        data.append({"rows": rows.astype(float), "cols": cols.astype(float),
                     "AoP": aop[yi, xi], "Q": stokes[0, yi, xi], "U": stokes[1, yi, xi],
                     "CompConfig": cfg.get("aop_compensation", {}), "Cpl": transform, "radius": radius,
                     "SunAz": np.radians(azimuth), "SunZen": np.radians(zenith),
                     "reference_heading_deg": record["reference_heading_deg"],
                     "timestamp_utc": timestamp.isoformat(), "image": record["image"],
                     "position_index": record.get("position_index", ""), "dolp_score": score})
    return data, np.array([d["reference_heading_deg"] for d in data]), shape


def compute_phi(alpha, gamma, aop, model_id):
    if model_id == 0:
        return np.arctan(np.cos(gamma)*np.tan(alpha-aop))
    if model_id == 1:
        return np.arctan(np.cos(gamma)*np.tan(alpha-(aop+np.pi/2)))
    if model_id == 2:
        return np.arctan(np.cos(gamma)*np.tan(alpha+aop))
    if model_id == 3:
        return np.arctan(np.cos(gamma)*np.tan(alpha+(aop+np.pi/2)))
    raise ValueError("model_id must be 0, 1, 2 or 3")


def compute_evectors(f, u0, v0, item, model_id=0):
    x, y = item["cols"]-v0, item["rows"]-u0
    mask = (x*x+y*y <= item["radius"]**2) & np.isfinite(item["AoP"])
    if f <= 0 or not np.any(mask):
        return np.empty((3, 0))
    x, y, aop = x[mask], y[mask], item["AoP"][mask]
    comp=item.get('CompConfig',{})
    if comp.get('enable_aop_model_comp',False):
        if 'Q' not in item or 'U' not in item:
            raise ValueError('Enabled AoP compensation requires Q/U samples')
        aop=correct_roi_aop(item['Q'][mask],item['U'][mask],comp)
    alpha, gamma = np.arctan2(x, y), np.arctan(np.hypot(x, y)/f)
    phi = compute_phi(alpha, gamma, aop, model_id)
    ca, sa, cg, sg, cp, sp = (np.cos(alpha), np.sin(alpha), np.cos(gamma),
                              np.sin(gamma), np.cos(phi), np.sin(phi))
    e_sensor = np.vstack((sa*cg*cp-ca*sp, ca*cg*cp+sa*sp, -sg*cp))
    return item["Cpl"] @ e_sensor


def estimate_heading(f, u0, v0, item, model_id=0):
    e = compute_evectors(f, u0, v0, item, model_id)
    if e.shape[1] < 2 or not np.isfinite(e).all():
        return None
    try:
        _, singular, vh = np.linalg.svd(e @ e.T)
    except np.linalg.LinAlgError:
        return None
    if singular[1]-singular[2] <= 1e-10*max(singular[0], 1.0):
        return None
    direction = vh[-1]
    if np.hypot(direction[0], direction[1]) <= 1e-10:
        return None
    # atan2(x,y) modulo pi preserves the original atan(x/y) residuals.
    haop = np.arctan2(direction[0], direction[1])
    return float(np.mod(np.degrees(item["SunAz"]-haop), 180.0))


def wrap180(angle_deg):
    return np.mod(np.asarray(angle_deg)+90.0, 180.0)-90.0


def frame_headings(params, data, model_id):
    values = [estimate_heading(*params, item, model_id) for item in data]
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    return np.asarray(values)


def absolute_metrics(headings, truth, cfg):
    raw_errors = wrap180(headings+cfg["evaluation_angle_offset_deg"]-truth)
    errors = wrap180(raw_errors-cfg['fixed_bias_deg'])
    return {"raw_rmse_deg": float(np.sqrt(np.mean(raw_errors**2))),
            "fixed_bias_corrected_rmse_deg": float(np.sqrt(np.mean(errors**2))),
            "mean_error_deg": float(np.mean(errors)), "std_error_deg": float(np.std(errors)),
            "max_abs_error_deg": float(np.max(np.abs(errors)))}


def write_outputs(mode, output_dir, cfg, data, params_before, params_after, summary):
    output = Path(output_dir).resolve() if output_dir else ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    before = frame_headings(params_before, data, cfg["model_id"])
    after = frame_headings(params_after, data, cfg["model_id"])
    if before is None or after is None:
        raise ValueError("Cannot export invalid heading outputs")
    truth = np.array([d["reference_heading_deg"] for d in data])
    summary.update(mode=mode, frame_count=len(data), initial_intrinsics_px=list(map(float, params_before)),
                   calibrated_intrinsics_px=list(map(float, params_after)),
                   extra_aop_compensation=bool(cfg.get('aop_compensation',{}).get('enable_aop_model_comp',False)),
                   before=absolute_metrics(before, truth, cfg), after=absolute_metrics(after, truth, cfg),
                   config_sha256=hashlib.sha256(cfg["config_path"].read_bytes()).hexdigest(),
                   sensor_calibration_sha256=hashlib.sha256(cfg["calibration_path"].read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256(cfg["manifest_path"].read_bytes()).hexdigest())
    summary['effective_config']={k:v for k,v in cfg.items() if not isinstance(v,Path)}
    summary['evaluation_convention']={k:cfg.get(k,0.) for k in
        ['evaluation_angle_offset_deg','fixed_bias_deg']}
    summary['evaluation_convention']['formula']='wrap180(heading + angle_offset - fixed_bias - reference)'
    bounds=np.asarray(cfg['bounds_px'],dtype=float)
    values=np.asarray(params_after,dtype=float)
    summary['parameter_boundary_diagnostics']={
        'fraction_of_search_range_to_nearest_bound':(np.minimum(values-bounds[:,0],bounds[:,1]-values)/(bounds[:,1]-bounds[:,0])).tolist(),
        'interpretation':'Small distance to a bound is a limitation of effective-parameter identification; convergence is not physical identifiability.'}
    summary['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['demo_common.py','aop_compensation.py','residual_weighting.py','focal_regularization.py','principal_direction_prior.py','guarded_direction_fit.py',
                     'calibration_rotation.py','calibration_static_heading_domain.py']}
    (output / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output / f"{mode}_frames.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["image", "timestamp_utc", "reference_heading_deg", "heading_before_deg",
                  "heading_after_deg", "error_before_deg", "error_after_deg"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(data):
            b = before[i]+cfg["evaluation_angle_offset_deg"]-cfg["fixed_bias_deg"]
            a = after[i]+cfg["evaluation_angle_offset_deg"]-cfg["fixed_bias_deg"]
            writer.writerow(dict(zip(fields, [item["image"], item["timestamp_utc"], truth[i],
                 b % 180, a % 180, float(wrap180(b-truth[i])), float(wrap180(a-truth[i]))])))
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Saved outputs to {output}", flush=True)
    return summary
