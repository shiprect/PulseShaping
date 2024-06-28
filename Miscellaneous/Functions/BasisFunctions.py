import numpy as np
from numpy import pi


def basis_k(N: int, k: int, n):
	"""
	Calculates the k-th DCT basis function for N samples.

	Parameters:
		N (int): The total number of samples and basis functions.
		k (int): The index of the basis function being used.
		n (numpy.ndarray): Array of sample indices ranging from 0 to N-1.

	Returns:
		numpy.ndarray: The k-th DCT basis function evaluated at each sample index in n.
	"""
	if k == 0:
		return np.array([1.0 / np.sqrt(N)] * N)
	else:
		return np.sqrt(2.0 / N) * np.cos((pi / N) * (n + 0.5) * k)


def basis_l(N: int, l: int, n):
	"""
	Calculates the l-th DCT basis function for N samples, using basis_k for computation.

	Parameters:
		N (int): The total number of samples and basis functions.
		l (int): The index of the basis function being used.
		n (numpy.ndarray): Array of sample indices ranging from 0 to N-1.

	Returns:
		numpy.ndarray: The l-th DCT basis function evaluated at each sample index in n.
	"""
	return basis_k(N, l, n)