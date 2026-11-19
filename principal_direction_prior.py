"""Fixed, weak principal-point axis prior (manuscript Sections 3.2--3.4).

All axes are modulo 180 degrees, measured from image row u toward column v.
The prior is only defined for level, zenith-looking observations. Its weight
has units deg^2 because the data term is a mean squared residual in degrees.
"""
import numpy as np


def image_solar_axes(data, initial_headings):
    """Convert the reconstructed level-frame solar axis into image u/v axes.

    The reconstruction uses sensor coordinates (x=column, y=row), followed
    by Cpl. Its level azimuth is atan2(sx,sy)=SunAz-heading. Undo Cpl before
    forming atan2(column,row). Horizontal directions suffice only if Cpl
    preserves the vertical axis; arbitrary tilt must not use this prior.
    """
    if initial_headings is None or len(initial_headings) != len(data):
        return None
    axes=[]
    for item, heading in zip(data, initial_headings):
        transform=np.asarray(item['Cpl'],dtype=float)
        if (transform.shape != (3,3) or not np.isfinite(transform).all()
                or not np.allclose(transform.T@transform,np.eye(3),atol=1e-10,rtol=0)
                or np.max(np.abs(transform[2,:2]))>1e-10
                or np.max(np.abs(transform[:2,2]))>1e-10
                or not np.isfinite(heading)):
            return None
        angle=item['SunAz']-np.radians(heading)
        sensor=transform.T@np.array([np.sin(angle),np.cos(angle),0.])
        axes.append(float(np.degrees(np.arctan2(sensor[0],sensor[1]))%180))
    return np.asarray(axes)


def disabled_state(initial, options, reason):
    return dict(requested=bool(options['enabled']),active=False,reason=reason,
                axis_deg=None,candidate_axis_deg=None,correlation=None,
                initial_intrinsics_px=list(map(float,initial)),
                weight_deg2=float(options['weight_deg2']),scale_px=float(options['scale_px']),
                scan_step_deg=float(options['scan_step_deg']),
                min_correlation=float(options['min_correlation']),min_pairs=int(options['min_pairs']),
                pair_count=0,scan_axes_deg=[],scan_correlations=[],
                initial_residual_magnitudes_deg=[],solar_axis_pairs_deg=[],
                axis_convention='atan2(column_offset, row_offset), modulo 180 deg')


def scan_axis(residual_magnitudes, solar_pairs, initial, options):
    """Scan paired |sin| differences; never recomputed inside optimization."""
    result=disabled_state(initial,options,'disabled_by_configuration')
    step=result['scan_step_deg']; scale=result['scale_px']; weight=result['weight_deg2']
    if (not np.isfinite([step,scale,weight,result['min_correlation']]).all()
            or not 0<step<180 or scale<=0 or weight<0
            or not -1<=result['min_correlation']<=1 or result['min_pairs']<4):
        raise ValueError('Invalid direction prior settings')
    if not result['requested']:
        return result
    y=np.asarray(residual_magnitudes,dtype=float)
    pairs=np.asarray(solar_pairs,dtype=float)
    if y.ndim!=1 or pairs.shape!=(len(y),2):
        raise ValueError('Prior requires one solar-axis pair per residual')
    result['pair_count']=len(y)
    if not np.isfinite(y).all() or not np.isfinite(pairs).all():
        result['reason']='invalid_initial_observations';return result
    result['initial_residual_magnitudes_deg']=y.tolist()
    result['solar_axis_pairs_deg']=pairs.tolist()
    if len(y)<result['min_pairs']:
        result['reason']='insufficient_pairs';return result
    if np.std(y)<1e-12:
        result['reason']='constant_residuals';return result
    best=None;best_corr=-np.inf
    for theta in np.arange(0.,180.,step):
        m=np.abs(np.sin(np.radians(pairs-theta)))
        g=np.abs(m[:,1]-m[:,0])
        corr=float(np.corrcoef(y,g)[0,1]) if np.std(g)>=1e-12 else None
        if corr is not None and not np.isfinite(corr):corr=None
        result['scan_axes_deg'].append(float(theta));result['scan_correlations'].append(corr)
        if corr is not None and corr>best_corr:
            best=float(theta);best_corr=corr
    if best is None:
        result['reason']='constant_predicted_pattern';return result
    result.update(candidate_axis_deg=best,correlation=best_corr)
    if best_corr<result['min_correlation']:
        result['reason']='low_correlation';return result
    if weight==0:
        result['reason']='zero_weight';return result
    result.update(active=True,reason='accepted',axis_deg=best)
    return result


def penalty(params, state):
    """Dimensionless squared perpendicular distance; no focal-length penalty."""
    if state is None or not state['active']:
        return 0.
    du,dv=np.asarray(params,dtype=float)[1:]-np.asarray(state['initial_intrinsics_px'])[1:]
    theta=np.radians(state['axis_deg'])
    return float(((-du*np.sin(theta)+dv*np.cos(theta))/state['scale_px'])**2)


def weighted_penalty(params,state):
    return 0. if state is None else state['weight_deg2']*penalty(params,state)
