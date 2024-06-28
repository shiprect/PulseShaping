import numpy as np
from functools import partial
from Miscellaneous.Functions.BasisFunctions import basis_k, basis_l


def DCT_VerifyBasis(N: int):
	"""
	Calculates and prints the inner product of all combinations of DCT basis functions for N samples.

	Parameters:
		N (int): The total number of samples and basis functions.

	Purpose:
		Computes the discrete cosine transform (DCT) for N samples across all N^2 combinations of basis functions.
		This is used to numerically verify the orthonormality of the DCT basis functions. Results are printed.
	"""
	# NOTE: Partial basis functions with values for the first 2 arguments are stored as lists to use
	#       for the later inner product calculations. I had originally used basic lambda functions
	#       which have a less obvious characteristic given to them by Python: Since the variables for
	#       the lambda function are evaluated after the for loop in the list compositions have
	#       completed, the odd-looking default assignments of "k=k" and "l=l" are actually required
	#       in order to achieve the desired and more obvious-looking result. I decided to switched to
	#       using functools.partial to acheive the same effect as that seems to be the more
	#       respectable method, but the lambda function method has been left posterity since it took
	#       me long enough to figure out through debugging.

	# bk = [lambda n, k=k: basis_k(N, k, n) for k in range(N)]
	# bl = [lambda n, l=l: basis_l(N, l, n) for l in range(N)]
	bk = [partial(basis_k, N, k) for k in range(N)]
	bl = [partial(basis_l, N, l) for l in range(N)]

	# NOTE: Compute the dot/inner product of each combination of basis functions.
	print(f'N: {N}')
	n_arr = np.arange(0, N)
	for k in n_arr:
		for l in n_arr:
			inner_product = lambda: bk[k](n_arr).dot(bl[l](n_arr))
			print(f'k: {k}, l: {l}, inner product: {inner_product()}')
	print()


def DCT(x_n, N: int, n: np.array):
	"""
	Calculates the Discrete Cosine Transform (DCT) of a given signal.

	Parameters:
		x_n (function): A function representing the input signal, which takes a numpy array of sample indices and returns the signal values at those indices.
		N (int): The total number of samples and basis functions.
		n (numpy.ndarray): Array of sample indices ranging from 0 to N-1.

	Returns:
		numpy.ndarray: The DCT coefficients of the input signal.

	Example:
		# Define the input signal as a function
		def x_n(n):
			return np.sin(2 * pi * n / 16)

		N = 16
		n = np.arange(0, N)

		# Calculate the DCT of the input signal
		Xc = DCT(x_n, N, n)
    """

	Xc = [sum(x_n(n) * basis_k(N, k, n)) for k in range(N)]
	return np.array(Xc)