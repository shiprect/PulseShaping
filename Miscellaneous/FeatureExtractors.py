import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib.pyplot as mplot


def maximumFrequency(filename, delta_t):
	"""
	Computes and plots the maximum frequency over time for an audio signal.

	Parameters:
		filename (str): Name of the audio file.
		delta_t (float): Time interval (in seconds) over which to compute the spectrogram segments.

	Returns:
		tuple:
			f (numpy.ndarray): Array of sample frequencies.
			t (numpy.ndarray): Array of segment times.
			S (numpy.ndarray): Spectrogram of the audio signal in dB, with time on the x-axis and frequency on the y-axis.

	Note:
		The function reads the audio file, computes its spectrogram, identifies the maximum frequency at each time segment, 
		and plots the maximum frequency curve over time. The spectrogram is also converted to dB scale and normalized by 
		subtracting its maximum value.
	"""
	rate, signal = wav.read(filename = filename)
	nperseg = int(rate * delta_t)
	f, t, S = sig.spectrogram(x = signal, fs = rate, nperseg = nperseg, noverlap = 0)
	# transposing so that time becomes x-axis of plot
	S = np.transpose(S)
	maxFreq = np.zeros(len(t))
	for i in range(len(t)):  # Iterate through t and find the maximum frequency for the bin
		maxFreq[i] = f[np.argmax(S[i], axis = 0)]
	# converting to dB
	S = 10 * np.log10(S)
	S -= S.max()

	print(maxFreq)
	# Plot Maximum Frequency Curve to verify
	mplot.plot(t, maxFreq)
	mplot.xlabel('$t$ (s)')
	mplot.ylabel('$f$ (Hz)')
	mplot.title(r'Maximum Frequency')
	mplot.tight_layout()
	return f, t, S
