import numpy as np

from Miscellaneous.LinearFrequencyModulation import lfm


def matched_filter(B, T):
	"""
	Creates a matched filter for a Linear Frequency Modulated (LFM) signal.

	Parameters:
		B (float): Bandwidth of the LFM signal in Hz.
		T (float): Duration of the LFM pulse in seconds.

	Returns:
		numpy.ndarray: The matched filter, which is the time-reversed and conjugated LFM signal.
	"""
	n, x1 = lfm(B, T)
	return np.conj(x1[::-1])
