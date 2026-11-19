"""Calibration-only selection of a weak direction prior.

This limits fitting loss; it does not guarantee unseen absolute-heading accuracy.
No external validation headings or absolute-error offsets are used. Callers may
provide an explicit extra parameter penalty; it is retained in every direction-prior ablation.
"""
from copy import deepcopy
from functools import lru_cache
import numpy as np
from scipy.optimize import differential_evolution
import principal_direction_prior as prior


def blocked_folds(count, mode):
    indices=np.arange(count)
    folds=[]
    for valid in np.array_split(indices,3):
        excluded=set(valid)
        if mode=='rotation':
            excluded.update(valid-1);excluded.update(valid+1)
        train=np.array([i for i in indices if i not in excluded],dtype=int)
        if len(valid)==0 or len(train)<4:
            raise ValueError('Insufficient pairs for three blocked folds')
        folds.append((train,valid))
    return folds


def effective_weight(max_weight, baseline_mse, baseline_penalty, fraction):
    values=np.array([max_weight,baseline_mse,baseline_penalty,fraction])
    if not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Weight inputs must be finite and nonnegative')
    return float(min(max_weight,fraction*baseline_mse/max(baseline_penalty,1e-12)))


def cv_accepts(baseline, candidate, counts, converged):
    a,b,n=map(lambda x:np.asarray(x,dtype=float),(baseline,candidate,counts))
    return bool(len(a)==3 and len(b)==3 and len(n)==3 and all(converged)
                and np.isfinite(a).all() and np.isfinite(b).all()
                and np.all(b<=1.05*a+1e-14)
                and np.sum(n*b)<=.99*np.sum(n*a))


def final_accepts(baseline_mse,candidate_mse,converged):
    return bool(converged and np.isfinite(candidate_mse)
                and candidate_mse<=1.01*baseline_mse+1e-14)


def fit_calibration(cfg, residual_function, direction, maxiter=None,
                    fixed_direction_prior=False, extra_penalty=None):
    """Return selected OptimizeResult, effective prior and all selection evidence."""
    initial=np.asarray(cfg['initial_intrinsics_px'],dtype=float)
    policy=cfg.get('prior_policy','guarded')
    if policy not in ('fixed','guarded'):
        raise ValueError('prior_policy must be fixed or guarded')
    fixed_direction_prior=fixed_direction_prior or policy=='fixed'
    if cfg.get('residual_weighting','equal')!='equal' and not fixed_direction_prior:
        raise ValueError('Covariance weighting requires fixed prior policy; blocked CV must rebuild whitening within each fold')
    settings=cfg['optimizer'].copy()
    method=settings.pop('method')
    if method!='differential_evolution':
        raise ValueError('Guarded fitting requires differential_evolution')
    if maxiter is not None:settings['maxiter']=maxiter
    if extra_penalty is not None and not fixed_direction_prior and cfg.get('lambda_reg',0)!=0:
        raise ValueError('Automatic direction selection requires lambda_reg=0')
    diagnostics=dict(policy='fixed' if fixed_direction_prior else 'blocked_cv_guard',
                     solver_settings=dict(method=method,**settings),folds=[],
                     max_prior_fraction_at_baseline=.1,min_cv_mse_improvement=.01,
                     max_fold_mse_increase=.05,max_full_data_mse_increase=.01,
                     uses_external_validation=False,uses_absolute_heading_loss=False)

    @lru_cache(maxsize=256)
    def cached(params):
        values=residual_function(np.asarray(params))
        return None if values is None else np.asarray(values,dtype=float)

    def residuals(x):return cached(tuple(map(float,x)))

    def mse(x,indices):
        values=residuals(x)
        if values is None or not np.isfinite(values).all():return 1e9
        return float(np.mean(values[indices]**2))

    initial_residuals=residuals(initial)
    if initial_residuals is None:raise ValueError('Invalid initial residuals')
    all_indices=np.arange(len(initial_residuals))
    runs=[]

    def solve(indices,state,start,label):
        def objective(x):
            loss=mse(x,indices)
            return loss+prior.weighted_penalty(x,state)+(0. if extra_penalty is None else extra_penalty(x))
        print(f'Fitting {label} ({len(indices)} residuals)...',flush=True)
        result=differential_evolution(objective,cfg['bounds_px'],x0=start,
            strategy='best1bin',mutation=(.5,1.),recombination=.7,workers=1,**settings)
        record=dict(label=label,success=bool(result.success),message=str(result.message),
                    parameters_px=result.x.tolist(),data_mse_deg2=mse(result.x,indices),
                    objective_deg2=float(result.fun),iterations=int(result.nit),
                    extra_parameter_penalty_deg2=0. if extra_penalty is None else float(extra_penalty(result.x)),
                    evaluations=int(result.nfev))
        runs.append(record)
        print(f"Finished {label}: success={result.success}; data RMS={np.sqrt(record['data_mse_deg2']):.6f}",flush=True)
        return result,record

    baseline,base_record=solve(all_indices,None,initial,'full_no_prior')
    diagnostics['baseline']=base_record
    selected=baseline
    state=deepcopy(direction)

    def disable(reason):
        state.update(active=False,reason=reason,weight_deg2=0.)
        diagnostics['decision']=reason

    if not direction['active']:
        disable(direction['reason'])
    elif fixed_direction_prior:
        selected,record=solve(all_indices,state,initial,'full_fixed_prior')
        diagnostics.update(decision='fixed_prior_requested',candidate=record)
    elif not baseline.success:
        disable('baseline_not_converged')
    else:
        try:folds=blocked_folds(len(all_indices),cfg['mode'])
        except ValueError:
            folds=[];disable('insufficient_pairs_for_blocked_validation')
        options=cfg['direction_prior']
        initial_y=np.asarray(direction['initial_residual_magnitudes_deg'])
        initial_pairs=np.asarray(direction['solar_axis_pairs_deg'])
        for number,(train,valid) in enumerate(folds):
            axis=prior.scan_axis(initial_y[train],initial_pairs[train],
                                 cfg['initial_intrinsics_px'],options)
            fit0,record0=solve(train,None,initial,f'fold_{number+1}_no_prior')
            axis['weight_deg2']=effective_weight(options['weight_deg2'],
                                record0['data_mse_deg2'],prior.penalty(fit0.x,axis),.1)
            if axis['active'] and axis['weight_deg2']>0 and fit0.success:
                fit1,record1=solve(train,axis,fit0.x,f'fold_{number+1}_weak_prior')
            else:fit1,record1=fit0,record0
            diagnostics['folds'].append(dict(train_indices=train.tolist(),validation_indices=valid.tolist(),
                direction_prior=axis,baseline=record0,candidate=record1,
                baseline_validation_mse_deg2=mse(fit0.x,valid),
                candidate_validation_mse_deg2=mse(fit1.x,valid),
                eligible=bool(axis['active'] and axis['weight_deg2']>0 and fit0.success and fit1.success)))
        reports=diagnostics['folds']
        accepted=cv_accepts([r['baseline_validation_mse_deg2'] for r in reports],
                            [r['candidate_validation_mse_deg2'] for r in reports],
                            [len(r['validation_indices']) for r in reports],
                            [r['eligible'] for r in reports])
        diagnostics['cv_accepted']=accepted
        if folds and not accepted: disable('blocked_validation_did_not_support_prior')
        elif accepted:
            state['weight_deg2']=effective_weight(options['weight_deg2'],
                     base_record['data_mse_deg2'],prior.penalty(baseline.x,state),.1)
            candidate,record=solve(all_indices,state,baseline.x,'full_weak_prior')
            diagnostics['candidate']=record
            if state['weight_deg2']>0 and final_accepts(base_record['data_mse_deg2'],record['data_mse_deg2'],candidate.success):
                selected=candidate
                state['reason']='accepted_by_blocked_validation_and_data_guard'
                diagnostics['decision']=state['reason']
            else:disable('full_fit_guard_rejected_prior')
    diagnostics['runs']=runs
    diagnostics['total_evaluations']=sum(r['evaluations'] for r in runs)
    return selected,state,diagnostics
