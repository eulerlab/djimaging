"""Joint (difference of) Gaussian fits of the sRFs of all color channels of a receptive field, with soft priors.

Compared to ``spatial_rf_utils.fit_rf_model`` (one channel at a time, no priors), ``fit_srfs_joint``

- puts a Gaussian prior on the center position (e.g. at the stimulus position of the recording field), which keeps fits
  of weak or noisy sRFs from wandering off to noise peaks at the border of the stimulus, and
- weakly couples the centers of the color channels, i.e. separations between channels are penalized relative to the
  size of their centers. A weak channel then follows the position of a strong one, but strong channels can still be
  offset from each other.

The Gaussian parametrization (``x_mean``, ``y_mean``, ``x_stddev``, ``y_stddev``, ``theta`` in pixels and radians) is
the one of ``astropy.modeling.functional_models.Gaussian2D``, i.e. the same as for ``fit_rf_model``, so that the
parameters can be used with e.g. ``plot_rf_utils.plot_srf_gauss_fit``.
"""
import itertools
import warnings

import numpy as np
from astropy.modeling.functional_models import Gaussian2D
from scipy.optimize import least_squares

GAUSS_PARAM_NAMES = ('amplitude', 'polarity', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta', 'r2')
DOG_PARAM_NAMES = GAUSS_PARAM_NAMES + (
    'amplitude_s', 'x_stddev_s', 'y_stddev_s', 'surround_ratio', 'surround_rel_amp')


def _unit_gauss2d(xx, yy, x0, y0, sx, sy, theta):
    return Gaussian2D.evaluate(xx, yy, 1., x0, y0, sx, sy, theta)


def _nan_result(shape, names):
    nan_arr = np.full(shape, np.nan)
    return nan_arr, nan_arr.copy(), nan_arr.copy(), [{k: np.nan for k in names} for _ in range(shape[0])]


def fit_srfs_joint(srfs: np.ndarray, kind: str = 'dog', prior_center: tuple[float, float] | None = None,
                   center_prior_sd: float | tuple[float, float] = 4., center_prior_weight: float = 0.05,
                   coupling_weight: float = 0.05, surround_ratio: tuple[float, float] = (2., 10.),
                   sigma_init: tuple[float, ...] = (1.5, 4.), max_nfev: int = 2000
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Fit a difference of Gaussians (or a Gaussian) to the sRFs of all channels jointly, with soft priors on the
    centers.

    Model per channel (pixel ``(i, j)`` at coordinate ``x=j, y=i``): ``amp * (G_c - w * G_s)`` with unit-peak
    Gaussians ``G``. The center ``G_c`` has its own position, angle and ``x_stddev, y_stddev``. The surround ``G_s`` is
    concentric with the center and has the same angle and aspect ratio, scaled by ``k`` in ``surround_ratio``.
    ``0 <= w <= 1``, so the surround always has the opposite sign of the center and the sRF has the center's sign at the
    center. ``amp`` is signed, i.e. the center can be the negative part of the sRF. For ``kind='gauss'`` the surround
    is omitted.

    The least-squares cost is the squared residual relative to the total sum of squares of all channels' sRFs (i.e. the
    fraction of variance unexplained, about the mean of zero) plus two soft penalties in the same units:

    - center prior: ``center_prior_weight * ((dx / sd_x) ** 2 + (dy / sd_y) ** 2)`` per channel, with ``(dx, dy)`` the
      offset of the channel's center from ``prior_center``;
    - coupling: ``coupling_weight * d ** 2 / (s_i ** 2 + s_j ** 2)`` per pair of channels, with ``d`` the distance
      between their centers and ``s = sqrt(x_stddev * y_stddev)`` their center sigmas, i.e. separations are penalized
      relative to the size.

    The fit is started from ``prior_center`` and from the largest |value| of the strongest channel, with each sigma in
    ``sigma_init``; the initial center sign of each channel is the sign of the sRF weighted with the initial center
    Gaussian. The start with the lowest penalized cost is kept.

    The lower bound of ``surround_ratio`` must not be close to 1: two similar Gaussians subtracted give a ring, i.e. the
    DoG can then fit a blob with a much larger center and a thin surround ring.

    Args:
        srfs: sRFs of shape ``(n_channels, ny, nx)`` or a single sRF ``(ny, nx)``.
        kind: ``'dog'`` or ``'gauss'``.
        prior_center: ``(x, y)`` of the center prior in pixels. Defaults to the center of the sRF.
        center_prior_sd: SD of the center prior in pixels, scalar or ``(sd_x, sd_y)``.
        center_prior_weight: Weight of the center prior (0: no prior).
        coupling_weight: Weight of the coupling between the channel centers (0: independent channels).
        surround_ratio: Bounds of the surround / center sigma ratio ``k``.
        sigma_init: Initial center sigmas (pixels) to start from.
        max_nfev: Maximum number of function evaluations per start.

    Returns:
        tuple: ``(srf_fits, srf_center_fits, srf_surround_fits, params)``, the arrays with the shape of ``srfs`` and
        ``params`` a list with a dict per channel with ``amplitude`` (signed center peak), ``polarity`` (center sign),
        ``x_mean, y_mean, x_stddev, y_stddev, theta`` of the center (pixels, radians) and the explained variance
        ``r2``; for the DoG also ``amplitude_s`` (signed surround peak), ``x_stddev_s, y_stddev_s``,
        ``surround_ratio`` (``k``) and ``surround_rel_amp`` (``w``). All NaN if the fit failed.
        For a single sRF ``(ny, nx)``, the arrays are 2d and ``params`` is a dict.
    """
    if kind not in ('dog', 'gauss'):
        raise ValueError(kind)
    if center_prior_weight > 0 and np.any(np.asarray(center_prior_sd) <= 0):
        raise ValueError(f'center_prior_sd must be positive, got {center_prior_sd}')

    srfs = np.asarray(srfs, dtype=float)
    single = srfs.ndim == 2
    if single:
        srfs = srfs[np.newaxis]
    if srfs.ndim != 3:
        raise ValueError(f'Expected sRFs of shape (n_channels, ny, nx), got {srfs.shape}')

    srf_fits, srf_center_fits, srf_surround_fits, params = _fit_srfs_joint(
        srfs=srfs, dog=kind == 'dog', prior_center=prior_center, center_prior_sd=center_prior_sd,
        center_prior_weight=center_prior_weight, coupling_weight=coupling_weight, surround_ratio=surround_ratio,
        sigma_init=sigma_init, max_nfev=max_nfev)

    if single:
        return srf_fits[0], srf_center_fits[0], srf_surround_fits[0], params[0]
    return srf_fits, srf_center_fits, srf_surround_fits, params


def _fit_srfs_joint(srfs, dog, prior_center, center_prior_sd, center_prior_weight, coupling_weight, surround_ratio,
                    sigma_init, max_nfev):
    names = DOG_PARAM_NAMES if dog else GAUSS_PARAM_NAMES
    n_ch, ny, nx = srfs.shape
    norm = np.sqrt(np.sum(srfs ** 2)) if np.all(np.isfinite(srfs)) else 0.
    if norm == 0:  # e.g. NaN sRFs of a failed STRF
        return _nan_result(srfs.shape, names)

    yy, xx = np.mgrid[:ny, :nx].astype(float)
    if prior_center is None:
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
    else:
        xc, yc = float(prior_center[0]), float(prior_center[1])
    prior_sd = np.broadcast_to(np.asarray(center_prior_sd, dtype=float), (2,))
    z = srfs / norm
    n_p = 8 if dog else 6  # x0, y0, theta, sx, sy, amp[, k, w]
    s_max = float(max(nx, ny))
    lb = np.tile([-0.5, -0.5, -np.inf, 0.3, 0.3, -np.inf] + ([surround_ratio[0], 0.] if dog else []), n_ch)
    ub = np.tile([nx - 0.5, ny - 0.5, np.inf, s_max, s_max, np.inf] + ([surround_ratio[1], 1.] if dog else []), n_ch)
    pairs = list(itertools.combinations(range(n_ch), 2))
    sqrt_wc, sqrt_wk = np.sqrt(center_prior_weight), np.sqrt(coupling_weight)

    def center_model(p):
        return p[5] * _unit_gauss2d(xx, yy, p[0], p[1], p[3], p[4], p[2])

    def surround_model(p):
        if not dog:
            return np.zeros_like(xx)
        return -p[5] * p[7] * _unit_gauss2d(xx, yy, p[0], p[1], p[6] * p[3], p[6] * p[4], p[2])

    def residuals(theta):
        ps = theta.reshape(n_ch, n_p)
        res = [(center_model(p) + surround_model(p) - z[c]).ravel() for c, p in enumerate(ps)]
        if center_prior_weight > 0:
            res.append(sqrt_wc * ((ps[:, :2] - [xc, yc]) / prior_sd).ravel())
        if coupling_weight > 0:
            for i, j in pairs:
                res.append(sqrt_wk * (ps[i, :2] - ps[j, :2]) / np.sqrt(ps[i, 3] * ps[i, 4] + ps[j, 3] * ps[j, 4]))
        return np.concatenate(res)

    ch0 = int(np.argmax(np.abs(z).max(axis=(1, 2))))
    iy, ix = np.unravel_index(np.argmax(np.abs(z[ch0])), (ny, nx))
    starts = sorted({(np.clip(xc, 0, nx - 1), np.clip(yc, 0, ny - 1)), (float(ix), float(iy))})
    best = None
    for (x0, y0), s0 in itertools.product(starts, sigma_init):
        g0 = _unit_gauss2d(xx, yy, x0, y0, s0, s0, 0.)
        p0 = np.concatenate([[x0, y0, 0., s0, s0, (np.sign(np.sum(g0 * z[c])) or 1.) * max(np.abs(z[c]).max(), 1e-3)]
                             + ([3., 0.2] if dog else []) for c in range(n_ch)])
        p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # With x_scale='jac', the default xtol stops the fit too early, e.g. at a too large surround
                sol = least_squares(residuals, p0, bounds=(lb, ub), x_scale='jac', xtol=1e-10, max_nfev=max_nfev)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        return _nan_result(srfs.shape, names)

    srf_center_fits, srf_surround_fits, params = np.zeros(srfs.shape), np.zeros(srfs.shape), []
    for c, p in enumerate(best.x.reshape(n_ch, n_p)):
        srf_center_fits[c] = center_model(p) * norm
        srf_surround_fits[c] = surround_model(p) * norm
        srf_fit_c = srf_center_fits[c] + srf_surround_fits[c]
        var_c = np.var(srfs[c])
        d = dict(amplitude=p[5] * norm, polarity=int(np.sign(p[5])), x_mean=p[0], y_mean=p[1],
                 x_stddev=p[3], y_stddev=p[4], theta=(p[2] + np.pi / 2) % np.pi - np.pi / 2,
                 r2=1. - np.var(srfs[c] - srf_fit_c) / var_c if var_c > 0 else np.nan)
        if dog:
            d.update(amplitude_s=-p[7] * p[5] * norm, x_stddev_s=p[6] * p[3], y_stddev_s=p[6] * p[4],
                     surround_ratio=p[6], surround_rel_amp=p[7])
        params.append(d)
    return srf_center_fits + srf_surround_fits, srf_center_fits, srf_surround_fits, params
