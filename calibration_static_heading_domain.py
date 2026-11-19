"""Time-separated static calibration adapted from the supplied research script.

Fits [f,u0,v0] using heading changes relative to a fixed reference frame and
an optional fixed direction prior. Initial-parameter regularization defaults
to zero; an explicit nonzero setting is a separate, non-manuscript variant.
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


def static_residuals(x, data, truth_headings_deg, model_id, ref_idx=0):
    if x[0] <= 0 or not 0 <= ref_idx < len(data):
        return None
    headings = frame_headings(x, data, model_id)
    truth = np.asarray(truth_headings_deg)
    if headings is None or not np.isfinite(truth).all():
        return None
    residuals = wrap180((headings-headings[ref_idx])-(truth-truth[ref_idx]))
    return np.delete(residuals, ref_idx)


def initial_parameter_penalty(x, init_guess, lambda_reg):
    if init_guess is None or lambda_reg <= 0:
        return 0.0
    return float(lambda_reg*np.sum(((np.asarray(x)-init_guess)/[100., 10., 10.])**2))


def heading_domain_cost(x, data, truth_headings_deg, model_id, init_guess=None,
                        lambda_reg=0.0, ref_idx=0, direction_prior=None):
    errors = static_residuals(x, data, truth_headings_deg, model_id, ref_idx)
    if errors is None or not len(errors):
        return 1e9
    return float(np.mean(errors**2)+initial_parameter_penalty(x, init_guess, lambda_reg)
                 +prior.weighted_penalty(x,direction_prior))


def estimate_direction_prior(cfg,data,truth,reference):
    initial=cfg['initial_intrinsics_px'];options=cfg['direction_prior']
    if not options['enabled']:
        return prior.disabled_state(initial,options,'disabled_by_configuration')
    h=frame_headings(initial,data,cfg['model_id'])
    if h is None:
        return prior.disabled_state(initial,options,'invalid_initial_observations')
    axes=prior.image_solar_axes(data,h)
    if axes is None:
        return prior.disabled_state(initial,options,'unsupported_geometry')
    errors=static_residuals(initial,data,truth,cfg['model_id'],reference)
    if errors is None:
        return prior.disabled_state(initial,options,'invalid_initial_observations')
    other=np.delete(axes,reference)
    pairs=np.column_stack((np.full(len(other),axes[reference]),other))
    return prior.scan_axis(np.abs(errors),pairs,initial,options)


def select_reference_frame(data, count=10):
    n = min(count, len(data))
    candidates = list(range(n)) if data[0]["SunZen"] > data[-1]["SunZen"] else list(range(len(data)-n, len(data)))
    scores = np.array([data[i]["dolp_score"] for i in candidates])
    if not np.isfinite(scores).all():
        raise ValueError("Invalid reference-frame DoLP scores")
    return candidates[int(np.argmax(scores))]


def run_demo(config_path=None, output_dir=None, maxiter=None, no_direction_prior=False,
             fixed_direction_prior=False, no_aop_compensation=False, paper_objective=False,no_focal_prior=False):
    start = time.perf_counter()
    cfg = read_config("static", config_path)
    cfg=focal.objective_config(cfg,paper_objective,no_focal_prior)
    if no_aop_compensation:cfg['aop_compensation']['enable_aop_model_comp']=False
    if no_direction_prior:cfg['direction_prior']['enabled']=False
    print("Loading bundled static images; AoP compensation="+str(cfg["aop_compensation"]["enable_aop_model_comp"]), flush=True)
    data, truth, shape = load_dataset(cfg)
    initial = np.asarray(cfg["initial_intrinsics_px"], dtype=float)
    reference = select_reference_frame(data, cfg["reference_candidate_count"])
    initial_errors = static_residuals(initial, data, truth, cfg["model_id"], reference)
    if initial_errors is None:
        raise ValueError("Initial parameters cannot reconstruct required observations")
    direction=estimate_direction_prior(cfg,data,truth,reference)
    print(f"Direction prior: {direction['reason']}; axis={direction['axis_deg']}; correlation={direction['correlation']}",flush=True)
    print(f"Loaded {len(data)} frames; reference={reference}; initial static RMS={np.sqrt(np.mean(initial_errors**2)):.6f} deg", flush=True)
    weighting=cfg.get('residual_weighting','equal')
    residual=lambda x:weight_residuals(static_residuals(x,data,truth,cfg['model_id'],reference),'static',weighting)
    result,direction,selection=fit_calibration(cfg,residual,
        direction,maxiter,fixed_direction_prior,
        extra_penalty=lambda x:initial_parameter_penalty(x,initial,cfg['lambda_reg'])+focal.penalty(x,cfg))
    final_errors = static_residuals(result.x, data, truth, cfg["model_id"], reference)
    if final_errors is None or not np.isfinite(result.fun) or result.fun >= 1e9:
        raise RuntimeError("Optimizer did not produce a feasible calibration")
    return write_outputs("static", output_dir, cfg, data, initial, result.x,
                         {"optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                          "optimizer_settings": selection['solver_settings'],
                          "prior_selection":selection,
                          "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                          "reference_frame_index": reference,
                          "reference_frame_image": data[reference]["image"],
                          "data_residual_rms_before_deg": float(np.sqrt(np.mean(initial_errors**2))),
                          "data_residual_rms_after_deg": float(np.sqrt(np.mean(final_errors**2))),
                          "residual_weighting":weighting,
                          "focal_regularization":focal.settings(cfg),
                          "focal_regularization_before_deg2":focal.penalty(initial,cfg),
                          "focal_regularization_after_deg2":focal.penalty(result.x,cfg),
                          "weighted_residual_rms_before_deg":float(np.sqrt(np.mean(residual(initial)**2))),
                          "weighted_residual_rms_after_deg":float(np.sqrt(np.mean(residual(result.x)**2))),
                          "regularization_after_deg2": initial_parameter_penalty(result.x, initial, cfg["lambda_reg"]),
                          "initial_regularization_weight":cfg['lambda_reg'],
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
    parser.add_argument('--no-aop-compensation',action='store_true',help='Disable empirical AoP correction for a separately refitted ablation')
    parser.add_argument('--paper-objective',action='store_true',help='Use equal relative residuals and disable focal regularization')
    parser.add_argument('--no-focal-prior',action='store_true',help='Disable the explicit initial-focal regularization')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--no-direction-prior',action='store_true',help='Disable direction prior only; retain configured focal setting')
    group.add_argument('--fixed-direction-prior',action='store_true',help='Use fixed weight without validation guard (ablation)')
    args = parser.parse_args()
    run_demo(args.config, args.output_dir, args.maxiter,args.no_direction_prior,args.fixed_direction_prior,args.no_aop_compensation,args.paper_objective,args.no_focal_prior)


if __name__ == "__main__":
    main()
