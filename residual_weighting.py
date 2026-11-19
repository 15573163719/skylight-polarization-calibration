"""Optional covariance weighting; the original residual definitions are unchanged.

Equal independent per-frame noise implies correlated differences. This optional
update uses 2*e.T@inv(D@D.T)@e / len(e), with the factor 2 retaining the marginal
difference-variance scale. The equal-weight paper objective remains available.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_triangular


@lru_cache(maxsize=16)
def _cholesky(count,mode):
    if mode=='rotation':
        matrix=2*np.eye(count)-np.eye(count,k=1)-np.eye(count,k=-1)
    elif mode=='static':
        matrix=np.eye(count)+np.ones((count,count))
    else:raise ValueError('Unknown calibration mode: '+mode)
    return np.linalg.cholesky(matrix)


def weight_residuals(values,mode,scheme='equal'):
    if mode not in ('rotation','static'):raise ValueError('Unknown calibration mode: '+mode)
    if scheme not in ('equal','difference_covariance'):raise ValueError('Unknown residual weighting: '+scheme)
    if values is None:return None
    e=np.asarray(values,dtype=float)
    if e.ndim!=1 or not len(e) or not np.isfinite(e).all():
        raise ValueError('Expected a nonempty finite residual vector')
    if scheme=='equal':return e
    return np.sqrt(2.)*solve_triangular(_cholesky(len(e),mode),e,lower=True,check_finite=False)
