"""Relative-rotation calibration adapted from the supplied research script.

Fits [f,u0,v0] against adjacent measured INS rotations. The example contains
14 consecutive positions, one frame each, with an optional fixed direction prior.
"""
import argparse
import time
import numpy as np
from guarded_direction_fit import fit_calibration
import principal_direction_prior as prior
from residual_weighting import weight_residuals
import focal_regularization as focal
from demo_common import (compute_phi, compute_evectors, estimate_heading, frame_headings,
                         load_dataset, read_config, wrap180, write_outputs)


def group_headings(headings_all, frames_per_pos):
    if len(headings_all) % frames_per_pos:
        raise ValueError("Incomplete turntable position: check frame manifest")
    result = []
    for start in range(0, len(headings_all), frames_per_pos):
        group = headings_all[start:start+frames_per_pos]
        if any(h is None or not np.isfinite(h) for h in group):
            return None
        angles = np.radians(np.asarray(group)*2)
        c, s = np.mean(np.cos(angles)), np.mean(np.sin(angles))
        if np.hypot(c, s) <= 1e-10:
            return None
        result.append(float(np.degrees(0.5*np.arctan2(s, c))))
    return result


def reference_steps(cfg, reference_headings):
    """Freeze adjacent reference angles independently of candidate intrinsics."""
    values = np.asarray(reference_headings, dtype=float)
    per_pos = cfg['frames_per_position']
    if values.ndim != 1 or not np.isfinite(values).all() or per_pos < 1:
        raise ValueError('Invalid reference headings or position grouping')
    means = group_headings(values, per_pos)
    if means is None or len(means) < 2:
        raise ValueError('At least two valid reference positions are required')
    kind = cfg.get('rotation_reference_kind', 'nominal_command')
    if kind == 'matched_ins':
        steps = wrap180(np.diff(means))
        if np.any(np.abs(steps) >= 90.-1e-8):
            raise ValueError('INS axial difference is ambiguous at 90 degrees')
    elif kind == 'nominal_command':
        steps = np.broadcast_to(np.asarray(cfg['rotation_step_deg'], dtype=float), (len(means)-1,)).copy()
    else:
        raise ValueError('Unknown rotation reference kind: '+str(kind))
    if not np.isfinite(steps).all():
        raise ValueError('Nonfinite rotation reference steps')
    return steps


def rotation_residuals(x, data, rotation_step_deg, frames_per_pos, model_id):
    if x[0] <= 0:
        return None
    headings = frame_headings(x, data, model_id)
    if headings is None:
        return None
    means = group_headings(headings, frames_per_pos)
    if means is None or len(means) < 2:
        return None
    return wrap180(np.diff(means)-rotation_step_deg)


def rotation_cost(x, data, rotation_step_deg, frames_per_pos, model_id, direction_prior=None):
    errors = rotation_residuals(x, data, rotation_step_deg, frames_per_pos, model_id)
    return 1e9 if errors is None else float(np.mean(errors**2)+prior.weighted_penalty(x,direction_prior))


def estimate_direction_prior(cfg,data,steps=None):
    initial=cfg['initial_intrinsics_px'];options=cfg['direction_prior']
    if not options['enabled']:
        return prior.disabled_state(initial,options,'disabled_by_configuration')
    h=frame_headings(initial,data,cfg['model_id'])
    if h is None:
        return prior.disabled_state(initial,options,'invalid_initial_observations')
    axes=prior.image_solar_axes(data,h)
    if axes is None:
        return prior.disabled_state(initial,options,'unsupported_geometry')
    grouped=group_headings(axes,cfg['frames_per_position'])
    if steps is None:
        steps = (reference_steps(cfg,[d['reference_heading_deg'] for d in data])
                 if cfg.get('rotation_reference_kind')=='matched_ins' else cfg['rotation_step_deg'])
    errors=rotation_residuals(initial,data,steps,cfg['frames_per_position'],cfg['model_id'])
    if grouped is None or errors is None:
        return prior.disabled_state(initial,options,'invalid_initial_observations')
    pairs=np.column_stack((grouped[:-1],grouped[1:]))
    return prior.scan_axis(np.abs(errors),pairs,initial,options)


def run_demo(config_path=None, output_dir=None, maxiter=None, no_direction_prior=False,
             fixed_direction_prior=False):
    start = time.perf_counter()
    cfg = read_config("rotation", config_path)
    focal.settings(cfg)
    if no_direction_prior:cfg['direction_prior']['enabled']=False
    print("Loading bundled rotation images; fixed AoP preprocessing and focal stabilization", flush=True)
    data, truth, shape = load_dataset(cfg)
    per_pos = cfg["frames_per_position"]
    for index, item in enumerate(data):
        if int(item["position_index"]) != index//per_pos:
            raise ValueError("Manifest positions must be consecutive and complete")
    initial = np.asarray(cfg["initial_intrinsics_px"], dtype=float)
    steps = reference_steps(cfg,truth)
    args = (data, steps, per_pos, cfg["model_id"])
    before = rotation_cost(initial, *args)
    direction=estimate_direction_prior(cfg,data,steps)
    print(f"Direction prior: {direction['reason']}; axis={direction['axis_deg']}; correlation={direction['correlation']}",flush=True)
    print(f"Loaded {len(data)} frames; initial rotation RMS={np.sqrt(before):.6f} deg", flush=True)
    weighting=cfg.get('residual_weighting','equal')
    residual=lambda x:weight_residuals(rotation_residuals(x,*args),'rotation',weighting)
    result,direction,selection=fit_calibration(cfg,residual,
                          direction,maxiter,fixed_direction_prior,extra_penalty=lambda x:focal.penalty(x,cfg))
    after = rotation_cost(result.x, *args)
    if not np.isfinite(after) or after >= 1e9:
        raise RuntimeError("Optimizer did not produce a feasible calibration")
    return write_outputs("rotation", output_dir, cfg, data, initial, result.x,
                         {"optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                          "optimizer_settings": selection['solver_settings'],
                          "prior_selection":selection,
                          "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                          "data_residual_rms_before_deg": float(np.sqrt(before)),
                          "data_residual_rms_after_deg": float(np.sqrt(after)),
                          "residual_weighting":weighting,
                          "focal_regularization":focal.settings(cfg),
                          "focal_regularization_before_deg2":focal.penalty(initial,cfg),
                          "focal_regularization_after_deg2":focal.penalty(result.x,cfg),
                          "rotation_reference_kind":cfg.get('rotation_reference_kind','nominal_command'),
                          "rotation_reference_steps_deg":steps.tolist(),
                          "weighted_residual_rms_before_deg":float(np.sqrt(np.mean(residual(initial)**2))),
                          "weighted_residual_rms_after_deg":float(np.sqrt(np.mean(residual(result.x)**2))),
                          "direction_prior":direction,
                          "direction_prior_loss_after":prior.penalty(result.x,direction),
                          "direction_prior_weighted_before_deg2":prior.weighted_penalty(initial,direction),
                          "direction_prior_weighted_after_deg2":prior.weighted_penalty(result.x,direction),
                          "objective_after_deg2": float(result.fun), "image_shape": list(shape),
                          "elapsed_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional configuration JSON")
    parser.add_argument("--output-dir", help="Output folder (default: demo/results)")
    parser.add_argument("--maxiter", type=int, help="Override optimizer iteration budget")
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--no-direction-prior',action='store_true',help='Disable direction prior only; retain configured focal setting')
    group.add_argument('--fixed-direction-prior',action='store_true',help='Use fixed weight without validation guard (ablation)')
    args = parser.parse_args()
    run_demo(args.config, args.output_dir, args.maxiter,args.no_direction_prior,args.fixed_direction_prior)


if __name__ == "__main__":
    main()
