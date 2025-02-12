import numpy as np
from scipy.optimize import curve_fit


def sigmoid(x, x0, k, a):
    """Define sigmoid function"""
    y = a / (1 + np.exp(-k * (x - x0)))
    return y


def init_sigmoid_params(x_data, y_data):
    """Initialize sigmoid parameters"""
    p0 = [x_data[np.argmax(y_data > (y_data.max() / 2))], 10., np.percentile(y_data, 90)]
    bounds = [(x_data[0], 1e-9, np.min(y_data)), (x_data[-1], np.inf, 1.2 * np.max(y_data))]
    return p0, bounds


def fit_sigmoid_with_retry(x_data, y_data, max_tries=3):
    p00, bounds = init_sigmoid_params(x_data, y_data)

    for i in range(max_tries):
        try:
            p0 = p00 if i == 0 else [np.random.uniform(a, np.minimum(b, 100)) for a, b in np.array(bounds).T]
            popt = curve_fit(sigmoid, x_data, y_data, p0=p0, bounds=bounds)[0]
            return popt
        except RuntimeError:
            pass
    return p00


def fit_sigmoid(y_data, x_data=None, sign=1, ax=None):
    """Fit sigmoid function to data, and estimate half amp"""
    np.random.seed(42)

    if x_data is None:
        x_data = np.arange(y_data.size)

    y_data = y_data * sign

    # Fit sigmoid curve to the data
    popt = fit_sigmoid_with_retry(x_data, y_data, max_tries=3)
    x0_fit, k_fit, a_fit = popt

    if (x0_fit > x_data.max()) | (x0_fit < x_data.min()):
        x0_fit = 0

    # Calculate half amplitude x value
    half_amplitude = a_fit / 2
    half_amplitude_x = x0_fit
    slope_at_half_amplitude = k_fit * half_amplitude * (1 - half_amplitude / a_fit)

    half_amplitude *= sign
    slope_at_half_amplitude *= sign

    if ax is not None:
        x_data_us = np.linspace(x_data[0], x_data[-1], x_data.size * 20)
        ax.scatter(x_data, y_data * sign, label='Data')
        ax.plot(x_data_us, sigmoid(x_data_us, *popt) * sign, 'r-',
                label='Fit: x0=%5.3f, k=%5.3f, A=%5.3f' % tuple(popt))
        ax.plot(half_amplitude_x, half_amplitude, 'gD', linestyle='--',
                label=f'x(Half Amplitude)={half_amplitude_x:.2f}')
        dt = np.min(np.diff(x_data))
        ax.plot([half_amplitude_x - 0.5 * dt, half_amplitude_x + 0.5 * dt],
                [half_amplitude - 0.5 * dt * slope_at_half_amplitude,
                 half_amplitude + 0.5 * dt * slope_at_half_amplitude],
                color='k', label=f'slope={slope_at_half_amplitude:.2f}')
        ax.legend(fontsize=6)

    return half_amplitude, half_amplitude_x, slope_at_half_amplitude
