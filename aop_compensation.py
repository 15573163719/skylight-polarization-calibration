"""Original empirical first-order AoP correction, with explicit JSON settings.

Coefficients are inherited, not fitted by the intrinsic calibration. See
README.md for their provenance and interpretation limitations.
"""
import numpy as np


def select_heading_model_profile(a_raw, cfg):
    mode=cfg['profile_mode'].lower()
    if mode=='auto':
        if a_raw<cfg['midday']['A_max']:return cfg['midday']
        if a_raw>cfg['evening']['A_min']:return cfg['evening']
        return cfg['normal']
    names={'normal':'normal','midday_low_a':'midday','evening_high_a':'evening'}
    if mode not in names:raise ValueError('Unknown AoP compensation profile: '+mode)
    return cfg[names[mode]]


def apply_aop_model_compensation(q_raw,u_raw,d_q,d_u,min_a,max_delta_deg,gain=1.,valid_mask=None):
    if valid_mask is None:valid_mask=np.ones_like(q_raw,dtype=bool)
    a_raw=np.hypot(q_raw,u_raw)
    theta_raw=.5*np.arctan2(u_raw,q_raw)
    valid=valid_mask & np.isfinite(q_raw) & np.isfinite(u_raw) & (a_raw>=min_a)
    delta=np.zeros_like(theta_raw,dtype=float)
    delta[valid]=(d_u*np.cos(2*theta_raw[valid])-d_q*np.sin(2*theta_raw[valid]))/(2*a_raw[valid])
    delta*=gain
    delta=np.clip(delta,-np.radians(max_delta_deg),np.radians(max_delta_deg))
    return theta_raw-delta,delta,valid


def correct_roi_aop(q,u,cfg):
    """Select a profile from the current ROI, then correct each pixel's AoP."""
    finite=np.isfinite(q)&np.isfinite(u)
    if not np.any(finite):raise ValueError('No finite Q/U values for AoP compensation')
    amplitude=np.hypot(np.mean(q[finite]),np.mean(u[finite]))
    profile=select_heading_model_profile(amplitude,cfg)
    return apply_aop_model_compensation(q,u,profile['dQ']*cfg['dS1_scale'],
        profile['dU']*cfg['dS2_scale'],profile['min_A'],profile['max_delta_deg'],
        profile.get('gain',1.),finite)[0]
