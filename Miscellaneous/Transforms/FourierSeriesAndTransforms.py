import numpy as np
from numpy import pi

from Miscellaneous.Functions.BasisFunctions import basis_k


def xH(t: np.ndarray, H: int, A: tuple, T: tuple) -> np.ndarray:
	"""
	Calculates the truncated Fourier series of a square wave using the H harmonics formula.

	Parameters:
		t (numpy.ndarray): Array of time steps (seconds).
		H (int): Integer specifying the range to sum over for the harmonics.
		A (tuple): Signal amplitude, given as a tuple where the first element is the amplitude value and the second element is the unit.
		T (tuple): Signal period, given as a tuple where the first element is the period value and the second element is the unit.

	Returns:
		numpy.ndarray: Truncated Fourier series approximation of the square wave at each time step in t.

	Example:
		# Parameters for xH(). These are pairs (2-tuples) with the first element being the quantity  and the second element
		being the unit.
			H = [3, 11, 101]  # limits for xH summation
			A = (1, 'a.u.')  # amplitude
			T = (0.50, 'sec')  # frequency = 2 Hz
			t = (np.linspace(-T[0], T[0], 1001), 'sec')  # Sampling every msec

		# Solution for xH(t). Each list element is a numpy array calculated for each value of H. Based on the Fourier
		series formula used, this would have the same arbitrary unit as A.
			xH_t = ([xH(t[0], h, A, T) for h in H], 'a.u.')
    """
	return sum(((2 * A[0]) / (1j * pi * n)) * np.exp((1j * 2 * pi * n * t) / T[0]) for n in range(-H, H + 1, 2))


def fourier_square_wave(T, A, H):
	"""
	Generates a time-domain signal representing an H-harmonic approximation of a square wave using a truncated Fourier series.

	Parameters:
		T (float): Period of the square wave (seconds).
		A (float): Amplitude of the square wave (arbitrary units).
		H (int): Number of harmonics to include in the approximation.

	Returns:
		tuple:
			t (numpy.ndarray): Array of time values over which the square wave is evaluated.
			x_H (numpy.ndarray): Real part of the H-harmonic approximation of the square wave.

	Example:
		T = 1.0  # Period of the square wave
		A = 1.0  # Amplitude of the square wave
		H = 10   # Number of harmonics

		t, x_H = fourier_square_wave(T, A, H)

		# Plot the result
		import matplotlib.pyplot as mplot

		mplot.plot(t, x_H)
		mplot.xlabel('Time (s)')
		mplot.ylabel('Amplitude')
		mplot.title('H-harmonic Approximation of a Square Wave')
		mplot.show()
	"""
	oversampling = 16  # used to make plots smooth
	Fs = oversampling * (2 * H/T)
	Ts = 1/Fs
	t = np.arange(-2*T, 2*T, Ts)
	x_H = np.zeros(len(t), dtype = np.complex128)
	for n in range(-H, H + 1):
		if n % 2:
			x_H += (2 * A/(1j * pi * n)) * np.exp(2j * pi * n * t/T)

	# since time-axis t is different for each value of H, return t and x_H also take real part because imaginary part
	# theoretically should be zero but isn't due to small round off errors, so just throw it away
	return t, x_H.real


def DFT(x_n, N: int, n: np.array):
	"""
	Calculates the Discrete Fourier Transform (DFT) of a given signal.

	Parameters:
		x_n (function): A lambda function representing the input signal x[n], which takes a numpy array of sample indices and returns the signal values at those indices.
		N (int): The total number of samples and basis functions.
		n (numpy.ndarray): Array of sample indices ranging from 0 to N-1.

	Returns:
		numpy.ndarray: The DFT coefficients of the input signal.

	Example:
		# Define the input signal as a lambda function
		x_n = lambda n: np.sin(2 * pi * n / 16)

		N = 16
		n = np.arange(0, N)

		# Calculate the DFT of the input signal
		Xf = DFT(x_n, N, n)
    """
	Xf = [sum(x_n(n) * basis_k(N, k, n)) for k in range(N)]
	return np.array(Xf)