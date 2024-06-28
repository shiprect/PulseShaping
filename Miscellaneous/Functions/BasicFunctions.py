import numpy as np


def rect(t):
	"""
	Calculates the rectangle function applied to an input array.

	Parameters:
	t (numpy.ndarray): Array of time steps from -1 to 1. The precision is determined by the sampling period (Ts).

	Returns:
	numpy.ndarray: Array with the rectangle function applied, where the function is 1 for |t| < 0.5 and 0 otherwise.
	"""
	x = np.zeros(len(t))
	x[abs(t) < 0.5] = 1
	return x


def Dirac_delta(t, sigma: float):
	"""
	Approximates the Dirac delta distribution using a Gaussian distribution.

	Parameters:
	t (numpy.ndarray): Array of time steps (seconds).
	sigma (float): Standard deviation of the Gaussian distribution used for approximation.

	Returns:
	numpy.ndarray: Array representing the approximate Dirac delta distribution.
	"""
	return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(t ** 2 / (2 * sigma ** 2)))
