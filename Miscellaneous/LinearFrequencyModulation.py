import numpy as np
from numpy import pi

from Miscellaneous.Functions.BasicFunctions import rect


def lfm(B, T):
	"""
	Generates a Linear Frequency Modulated (LFM) signal.

	Parameters:
		B (float): Bandwidth of the LFM signal in Hz.
		T (float): Duration of the LFM pulse in seconds.

	Returns:
		tuple: A tuple containing:
			- n (numpy.ndarray): Array of time indices.
			- numpy.ndarray: The complex LFM signal.
	"""
	Fs = B
	Ts = 1 / Fs
	N = int(round(T / Ts))
	n = np.arange(-N / 2, N / 2)

	return n, np.exp(1j * pi * B / T * (n * Ts) ** 2)


def LFMpulse(t, ui: complex, Bi: float, Ti: float):
	"""
	Returns a time-reversed and complex conjugated LFM pulse signal over a given time array.

	Parameters:
		t (numpy.ndarray): Array of time steps over which the LFM signal is applied.
		ui (complex): Indicates the direction of the pulse sweep (1j for increasing, -1j for decreasing with time).
		Bi (float): Bandwidth of the LFM signal in Hz.
		Ti (float): Duration of the LFM pulse in seconds.

	Returns:
		numpy.ndarray: The non-zero values of the LFM pulse in its time-reversed and complex conjugated form.
	"""
	ki = (Bi / Ti, 'Hz/sec')  # Sweep rate
	h = rect(-t / Ti) * np.exp(ui * ki[0] * (-t) ** 2)
	return h[h.nonzero()[0][0]:h[h.nonzero()[0][-1]].conj()[::-1]]
