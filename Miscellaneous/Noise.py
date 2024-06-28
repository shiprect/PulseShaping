import numpy as np


def awgn(SNR, N):
	"""
	Generates additive white Gaussian noise (AWGN) for a given signal-to-noise ratio (SNR) and number of samples.

	Parameters:
		SNR (float): Signal-to-noise ratio. The power ratio between the signal and the noise.
		N (int): Number of samples of the noise to generate.

	Returns:
		numpy.ndarray: Array of complex AWGN samples with zero mean and the specified noise power.

	Note:
	The generated noise has a standard deviation of sqrt(noise_power / 2) for both real and imaginary parts, ensuring the
	total noise power matches the specified SNR.
    """
	sig_power = 1
	noise_power = sig_power / SNR
	return np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))