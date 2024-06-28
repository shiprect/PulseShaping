import numpy as np
import matplotlib.pyplot as mplot
import scipy.io.wavfile as wav
import scipy.signal as sig


def plot_image(
		x, y, z, figsize = None, label = None,
		interpolation = 'none',
		cmap = mplot.cm.jet,
		cticks = None,
		):
	# figsize is in units of inches

	dx = np.median(np.diff(np.sort(np.unique(x))))
	dy = np.median(np.diff(np.sort(np.unique(y))))

	if type(figsize) != type(None):
		fig = mplot.figure(figsize = figsize)
	else:
		fig = mplot.figure()

	mplot.imshow(
			z.T,
			aspect = 'auto',
			origin = 'lower',
			interpolation = interpolation,
			extent = (
					x.min() - dx / 2.,
					x.max() + dx / 2.,
					y.min() - dy / 2.,
					y.max() + dy / 2.,
					),
			cmap = cmap,
			)

	if cticks is None:
		cb = mplot.colorbar()
	else:
		cb = mplot.colorbar(ticks = cticks)

	if label != None:
		cb.set_label(label)

	return cb


def main():
	filename = 'directory'

	rate, signal = wav.read(filename = filename)

	time_per_chunk = 0.1
	nperseg = int(rate * time_per_chunk)
	f, t, S = sig.spectrogram(x = signal, fs = rate, nperseg = nperseg, noverlap = 0)

	# transposing so that time becomes x-axis of plot
	S = np.transpose(S)

	# converting to dB
	S = 10 * np.log10(S)
	S -= S.max()

	plot_image(
		t, f, S,
		figsize = (14, 6),
		label = 'Spectrogram (dB/Hz)', cmap = mplot.cm.magma
		)
	mplot.ylim(0, 800)  # zooming in on freq axis to see tones better
	mplot.xlabel('$t$ (s)')
	mplot.ylabel('$f$ (Hz)')
	mplot.title(r'$S(t,f)$ of %s' % filename + '\n$\Delta t = %g$ s' % time_per_chunk)
	mplot.clim(-20, 0)
	mplot.tight_layout()

	mplot.show()


if __name__ == '__main__':
	main()