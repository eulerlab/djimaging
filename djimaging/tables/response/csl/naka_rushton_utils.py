import warnings

import numpy as np
from scipy.optimize import curve_fit


def naka_rushton(S: np.ndarray, R_base: float, R_max: float, S50: float, n: float):
    """
    Naka-Rushton function.

    Args:
        S (np.ndarray): Stimulus intensity.
        R_base (float): Baseline response.
        R_max (float): Maximum response.
        S50 (float): Half-maximum stimulus.
        n (float): Slope parameter.

    Returns:
        np.ndarray: Response values.
    """
    if S50 == 0.0:
        return np.full_like(S, R_max * 0.5 + R_base)
    nr = np.zeros_like(S)
    non_zero = S != R_base
    nr[non_zero] = R_max * (S[non_zero] ** n) / (S[non_zero] ** n + S50 ** n) + R_base
    return nr


def init_naka_rushton(x_data: np.ndarray, y_data: np.ndarray, noise=False):
    """
    Initialize parameters for the Naka-Rushton function.

    Args:
        x_data (np.ndarray): Array of stimulus intensity values.
        y_data (np.ndarray): Array of response values.
        noise (bool): If True, adds random perturbations to initial estimates.

    Returns:
        tuple: Initial estimates for R_base, R_max, S50, and n.
    """
    # Base parameter estimates
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_rng = x_max - x_min
    y_min, y_max = np.min(y_data), np.max(y_data)
    y_rng = y_max - y_min

    R_base_initial = y_min
    R_max_initial = 0.9 * y_max
    R_rng = R_max_initial - R_base_initial
    S50_initial = x_data[np.argmin(np.abs(y_data - R_rng / 2))]
    n_initial = y_rng / x_rng  # Default slope guess

    # Add noise to the estimates if requested
    if noise:
        R_base_initial = R_base_initial + np.ptp(y_data) * 0.1 * (np.random.rand() - 0.5)
        R_max_initial = R_max_initial * (1 + 0.1 * (np.random.rand() - 0.5))
        S50_initial = S50_initial * (1 + 0.1 * (np.random.rand() - 0.5))
        n_initial = n_initial * (1 + 0.2 * (np.random.rand() - 0.5))

    p_bounds = [(y_min, y_min, np.nextafter(x_min, x_max), 0),
                (y_max, np.inf, np.inf, 100 * y_rng / x_rng)]

    # Ensure within bounds
    R_base_initial = min(max(p_bounds[0][0], R_base_initial), p_bounds[1][0])  # Ensure non-negative
    R_max_initial = min(max(p_bounds[0][1], R_max_initial), p_bounds[1][1])
    S50_initial = min(max(p_bounds[0][2], S50_initial), p_bounds[1][2])
    n_initial = min(max(p_bounds[0][3], n_initial), p_bounds[1][3])

    p_initial = R_base_initial, R_max_initial, S50_initial, n_initial

    return p_initial, p_bounds


def fit_naka_rushton_repeated(x_data: np.ndarray, y_data: np.ndarray, n: int = 3):
    """
    Fit Naka-Rushton function to data with retries.

    Args:
        x_data (array-like): Stimulus data.
        y_data (array-like): Response data.
        n (int): Number of retries.

    Returns:
        tuple: Optimal parameters for R_base, R_max, S50, and n.
    """
    np.random.seed(42)
    p_initial, p_bounds = init_naka_rushton(x_data, y_data, noise=False)

    popt_best = p_initial
    mse_best = np.inf

    for i in range(n):
        p_initial_ = p_initial if i == 0 else init_naka_rushton(x_data, y_data, noise=True)[0]

        try:
            popt = curve_fit(naka_rushton, x_data, y_data, p0=p_initial_, bounds=p_bounds)[0]
            y_fit = naka_rushton(x_data, *popt)
            mse = np.mean((y_data - y_fit) ** 2)

            if mse < mse_best:
                popt_best = popt
                mse_best = mse
        except RuntimeError:
            pass

    return popt_best


def fit_naka_rushton(x_data: np.ndarray, y_data: np.ndarray, ax=None):
    """
    Fit Naka-Rushton function to data and estimate half-maximum metrics.
    This function will use a linear fit if the Naka-Rushton fit is not good.
    The function will also estimate the slope at the half-maximum point; the half-maximum point is the point
    where the response is halfway between the minimum and maximum response for the given input range.
    This is different from the standard usage of the Naka-Rushton function, but more stable for non-saturating data.

    Args:
        x_data (array-like): Stimulus data.
        y_data (array-like): Response data.
        ax (matplotlib.axes.Axes, optional): Axis for plotting. Defaults to None.

    Returns:
        tuple: Half-maximum response, corresponding stimulus, and slope at that point.
    """
    np.random.seed(42)

    popt = fit_naka_rushton_repeated(x_data, y_data, n=3)
    R0_fit, Rm_fit, S50_fit, n_fit = popt

    # Compare MSE of linear fit
    linear_fit = np.polyfit(x_data, y_data, 1)
    linear_fit_fn = np.poly1d(linear_fit)

    y_fit_linear = linear_fit_fn(x_data)
    y_fit_naka_rushton = naka_rushton(x_data, *popt)

    mse_linear = np.mean((y_data - y_fit_linear) ** 2)
    mse_naka_rushton = np.mean((y_data - y_fit_naka_rushton) ** 2)

    # Estimate half-maximum stimulus and slope at that point
    y_fit_min, y_fit_max = np.min(y_fit_naka_rushton), np.max(y_fit_naka_rushton)
    half_max = (y_fit_min + y_fit_max) / 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        half_max_stimulus = (-((R0_fit - half_max) * S50_fit ** n_fit) / (R0_fit - half_max + Rm_fit)) ** (1 / n_fit)
        slope_at_half_max = (
                (n_fit * Rm_fit * half_max_stimulus ** (-1 + n_fit) * S50_fit ** n_fit) /
                (half_max_stimulus ** n_fit + S50_fit ** n_fit) ** 2
        )

    if (not np.isfinite(half_max_stimulus)) or (not np.isfinite(slope_at_half_max)):
        use_linear = True
    elif mse_naka_rushton > mse_linear * 1.05:  # 5% tolerance, e.g. if Naka-Rushton is almost linear but a good fit
        use_linear = True
    else:
        use_linear = False

    if use_linear:
        y_fit_min, y_fit_max = np.min(y_fit_linear), np.max(y_fit_linear)
        half_max = (y_fit_min + y_fit_max) / 2
        half_max_stimulus = np.nan
        slope_at_half_max = linear_fit[0]

    # Optional plotting
    if ax is not None:
        x_data_us = np.linspace(x_data[0], x_data[-1], x_data.size * 20)
        ax.scatter(x_data, y_data, label='Data', color='k')
        ax.plot(x_data_us, naka_rushton(x_data_us, *popt), 'r-',
                label=f'NR Fit: MSE={mse_naka_rushton:.3f}, R_max={Rm_fit:.3f}, S50={S50_fit:.3f}, n={n_fit:.3f}, R_base={R0_fit:.3f}.')
        ax.plot(x_data, linear_fit_fn(x_data), 'b-',
                label=f'Linear Fit: MSE={mse_linear:.3f}, slope={linear_fit[0]:.3f}, intercept={linear_fit[1]:.3f}')
        ax.plot(half_max_stimulus, half_max, 'gD', label=f'S(Half-Max)={half_max_stimulus:.2f}')
        dt = np.min(np.diff(x_data))
        ax.plot([half_max_stimulus - 0.5 * dt, half_max_stimulus + 0.5 * dt],
                [half_max - 0.5 * dt * slope_at_half_max,
                 half_max + 0.5 * dt * slope_at_half_max],
                color='lime', label=f'slope={slope_at_half_max:.2f}')
        ax.axhline(y_fit_min, color='black', linestyle='--', alpha=0.4)
        ax.axhline(y_fit_max, color='black', linestyle='--', alpha=0.4)
        ax.axhline(0.5 * (y_fit_min + y_fit_max), color='black', linestyle='--', alpha=0.8)

        ax.annotate('Linear fit used' if use_linear else 'Naka-Rushton fit used', xy=(0.5, 0.2),
                    xycoords='axes fraction', ha='center')

        ax.legend(fontsize=6)

    return half_max, half_max_stimulus, slope_at_half_max
