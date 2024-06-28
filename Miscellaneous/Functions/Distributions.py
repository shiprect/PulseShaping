import numpy as np


def delta_gaussian(sigma):
    """
    Approximates the Dirac delta distribution using a Gaussian (bell curve).

    Parameters:
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        tuple:
            t (numpy.ndarray): Array of time steps from -1 to 1 with a step size of sigma/10.
            x (numpy.ndarray): Gaussian approximation of the Dirac delta distribution at each time step in t.

    Note:
        The function creates a Gaussian distribution centered at zero with a standard deviation of sigma, and computes its
        values over the interval from -1 to 1 with a step size of sigma/10.
    """
    Ts = sigma / 10
    t = np.arange(-1, 1, Ts)
    x = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))

    return t, x