"""Synthetic linear-RF data for tests. Ported from the `rfest.simulate` module of the RFEst dev branch."""

import numpy as np
import scipy.stats


def uvec(x):
    return x / np.linalg.norm(x)


def gaussian2d(dims=(25, 25), std=(2., 2.), mean=(0, 0)):
    """Normalized 2D gaussian."""
    x = np.arange(0, dims[0]) - dims[0] / 2. + 0.5
    y = np.arange(0, dims[1]) - dims[1] / 2. + 0.5
    xx, yy = np.meshgrid(x, y, indexing='ij')
    rf = scipy.stats.multivariate_normal(mean=mean, cov=np.diag(std) ** 2).pdf(
        np.vstack([xx.flat, yy.flat]).T).reshape(dims)
    return uvec(rf)


def rf_to_3d(trf, srf):
    """Normalized space-time separable RF, shape (nt, ny, nx)."""
    return uvec(np.kron(trf, srf.flat).reshape(trf.shape + srf.shape))


def generate_data_3d_stim(rf_kind='gauss', response_noise='none', stim_noise='white', design_matrix=True,
                          n_stim_frames=1000, n_reps_per_frame=1, shift=0, baseline=1.):
    """Simulate a linear response to 2D noise. Returns (w_true, X or stim, y, dt, dims)."""
    from rfest import build_design_matrix

    if stim_noise != 'white':
        raise NotImplementedError(stim_noise)

    if rf_kind == 'gauss':
        srf = gaussian2d(dims=(10, 8), std=(2., 2.))
    else:
        raise NotImplementedError(rf_kind)

    trf = np.array([0, -0.2, -0.3, -0.5, -1, -0.5, 0.1, 1.0, 0.5, 0.0])
    trf = np.concatenate([trf, np.zeros(shift)])
    w_true = rf_to_3d(trf, srf)

    dims = w_true.shape
    dt = 1.

    stim = np.random.randn(n_stim_frames, *dims[1:])
    stim = np.repeat(stim, n_reps_per_frame, axis=0)

    X = build_design_matrix(stim, nlag=dims[0], shift=shift)
    # Positive baseline: the STA is normalized by sum(y), so its sign is only well defined for a mostly positive
    # response (e.g. spike counts). Without it the sign of the estimate depends on the random seed.
    y = X @ w_true.flatten() + baseline

    if response_noise == 'gaussian':
        y = np.random.normal(y)
    elif response_noise != 'none':
        raise NotImplementedError(response_noise)

    assert X.shape[0] == y.size

    return w_true, (X if design_matrix else stim), y, dt, dims
