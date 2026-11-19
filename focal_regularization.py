"""Explicit initial-focal regularization, separate from the original objectives.

The anchor is a declared initial input, not a manuscript fitted result. The
normalization scale and weight are design choices, not measured uncertainty.
"""
import numpy as np


def settings(cfg):
    options=cfg['focal_regularization']
    explicit='anchor_px' in options
    anchor=float(options['anchor_px'] if explicit else cfg['initial_intrinsics_px'][0])
    source=options.get('anchor_source','') if explicit else 'Optimization initial focal length; no measured uncertainty claimed'
    if not isinstance(source,str) or not source.strip():
        raise ValueError('An explicit focal anchor requires its source to be stated')
    weight=float(options['weight_deg2']);fraction=float(options['scale_fraction'])
    if not np.isfinite([anchor,weight,fraction]).all() or anchor<=0 or weight<0 or fraction<=0:
        raise ValueError('Focal anchor and scale must be finite positive; weight must be finite nonnegative')
    if cfg.get('prior_policy','fixed')!='fixed':
        raise ValueError('Focal regularization currently requires the fixed direction-prior policy')
    return dict(enabled=True,anchor_px=anchor,scale_px=anchor*fraction,scale_fraction=fraction,
                weight_deg2=weight,source=source,anchor_kind='explicit_config' if explicit else 'optimization_initial',
                anchor_independently_verified=False,
                measured_uncertainty_claimed=False)


def penalty(params,cfg):
    state=settings(cfg)
    f=float(params[0])
    if not np.isfinite(f):raise ValueError('Nonfinite candidate focal length')
    return float(state['weight_deg2']*((f-state['anchor_px'])/state['scale_px'])**2)


