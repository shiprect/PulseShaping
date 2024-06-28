import numpy as np
import matplotlib.pyplot as mplot
import scipy.signal as sig
import scipy.io.wavfile as wav


def plot_image(x, y, z, figsize = None, label = None, interpolation = 'none', cmap = mplot.cm.jet, cticks = None):
	# figsize is in units of inches
	dx = np.median(np.diff(np.sort(np.unique(x))))
	dy = np.median(np.diff(np.sort(np.unique(y))))

	if type(figsize) != type(None):
		fig = mplot.figure(figsize=figsize)
	else:
		fig = mplot.figure()

	mplot.imshow(
			z.T,
			aspect='auto',
			origin='lower',
			interpolation=interpolation,
			extent=(
					x.min() - dx/2.,
					x.max() + dx/2.,
					y.min() - dy/2.,
					y.max() + dy/2.,
					),
			cmap=cmap,
			)

	if cticks is None:
		cb = mplot.colorbar()
	else:
		cb = mplot.colorbar(ticks=cticks)

	if label != None:
		cb.set_label(label)

	return cb


def wvd(x, fs, nperseg, noverlap):
	# time axes
	N = len(x)
	mu = np.arange(0, N, nperseg - noverlap)
	tau = np.arange(-2*N + 1, 2*N)

	# filling up kernel array
	Kx = np.zeros((len(mu), len(tau)), dtype=np.complex128)
	for n, tauv in enumerate(tau):
		left = rollzeros(x, (-tauv)//2)[mu]
		right = np.conj(rollzeros(x, tauv//2)[mu])
		Kx[:,n] = left*right
	print('done filling up kernel array')

	# taking FFT along tau axis
	Wx = np.fft.fftshift(np.fft.fft(Kx, axis=1), axes=1)/fs
	print('done with FFT')

	f = np.fft.fftshift(np.fft.fftfreq(len(tau)))*fs

	# fixing phase shift to account for window being shifted to right
	shifts = np.zeros(Kx.shape, dtype=np.complex128)
	L = min(tau)
	for n in range(Kx.shape[0]):
		shifts[n,:] = np.exp(-2j*np.pi*f/fs*L)
	Wx *= shifts

	return f, mu/fs, Wx.real


def rollzeros(x, n):
	xr = np.roll(x, n)
	if n >= 0:
		xr[:n] = 0
	else:
		xr[n:] = 0
	return xr


def main():
	# generating chirp
	Fs = 6000
	B = 1e3
	T = 1
	tt = np.arange(-1*T, 1*T, 1/Fs)
	std = int(round(Fs*T/8))
	fc = 2.e3
	w = sig.windows.gaussian(len(tt), std)
	x = np.cos(2*np.pi*fc*tt + np.pi*B/T*tt**2)*w
	z = np.exp(1j*(2*np.pi*fc*tt + np.pi*B/T*tt**2))*w
	N0 = 1e-9 # adding noise
	x += np.random.randn(len(tt))*np.sqrt(N0)
	z += np.random.randn(len(tt))*np.sqrt(N0)
	print('expected noise floor:', 10*np.log10(N0/Fs), 'dBV/Hz')
	wav.write('audiochirp.wav', Fs, x)

	# computing optimal window length
	Delta = np.sqrt(2*T/B)
	Delta_samps = int(round(Delta*Fs))

	# window parameters
	nperseg = Delta_samps
	nfft = nperseg*2
	noverlap = nperseg*95//100

	# forming spectrogram
	f, t, Xw = sig.stft(x,
		fs=Fs,
		nperseg=nperseg,
		noverlap=noverlap,
		nfft=nfft,
		scaling='psd',
		return_onesided=False,
		)
	Sx = abs(Xw)**2

	# transposing so time is x-axis
	Sx = Sx.T

	# need to fftshift so image looks right
	f = np.fft.fftshift(f)
	Sx = np.fft.fftshift(Sx, axes=1)

	Sx = 10*np.log10(abs(Sx))
	Sx -= Sx.max()

	scale = 0.5
	cmin = -60

	figsize = scale*np.array([18, 9])
	plot_image(t, f, Sx, figsize=figsize, cmap=mplot.cm.magma, label='PSD (dB/Hz)')
	mplot.ylim(0, Fs / 2)
	mplot.clim(cmin, 0)
	mplot.xlabel(r'$t$ (s)')
	mplot.ylabel(r'$f$ (Hz)')
	mplot.title(r'$S_x^w(t, f)$')
	mplot.tight_layout()
	mplot.savefig('specgram.pdf')

	# computing WD
	f, t, Wx = wvd(x, fs=Fs, nperseg=nperseg, noverlap=noverlap)
	Wx = 10*np.log10(abs(Wx))
	Wx -= Wx.max()

	plot_image(t, f, Wx, figsize=figsize, cmap=mplot.cm.magma,
		label='PSD (dB/Hz)')
	mplot.ylim(0, Fs / 2)
	mplot.clim(cmin, 0)
	mplot.xlabel(r'$t$ (s)')
	mplot.ylabel(r'$f$ (Hz)')
	mplot.title(r'$|W_x(t, f)|$')
	mplot.tight_layout()
	mplot.savefig('wd.pdf')

	# computing WVD
	#z = sig.hilbert(x)
	f, t, Wz = wvd(z, fs=Fs, nperseg=nperseg, noverlap=noverlap)
	Wz = 10*np.log10(abs(Wz))
	Wz -= Wz.max()

	plot_image(t, f, Wz, figsize=figsize, cmap=mplot.cm.magma,
		label='PSD (dB/Hz)')
	mplot.ylim(0, Fs / 2)
	mplot.clim(cmin, 0)
	mplot.xlabel(r'$t$ (s)')
	mplot.ylabel(r'$f$ (Hz)')
	mplot.title(r'$|W_z(t, f)|$')
	mplot.tight_layout()
	mplot.savefig('wd_complex.pdf')

	mplot.show()


if __name__ == '__main__':
	main()
