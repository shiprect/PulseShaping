import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "cm"
import scipy.signal as sig
import scipy.io.wavfile as wav


def plot_image(x, y, z, figsize=None, label=None,
        interpolation='none',
        cmap=plt.cm.jet,
        cticks=None,
        ):
    # figsize is in units of inches

    dx = np.median(np.diff(np.sort(np.unique(x))))
    dy = np.median(np.diff(np.sort(np.unique(y))))

    if type(figsize) != type(None):
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    plt.imshow(
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
        cb = plt.colorbar()
    else:
        cb = plt.colorbar(ticks=cticks)

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
    B = 50
    T = 1
    tt = np.arange(-0.6*T, 0.6*T, 1/Fs)
    std = int(round(Fs*T/6))
    fc = 25
    w = sig.windows.gaussian(len(tt), std)
    x = np.cos(2*np.pi*fc*tt + np.pi*B/T*tt**2)*w
    N0 = 1e-9 # adding noise
    x += np.random.randn(len(tt))*np.sqrt(N0)


    z = sig.hilbert(x)

    scale = 0.5
    figsize = scale*np.array([18, 9])

    plt.figure(figsize=figsize)
    plt.plot(tt, x, label=r'$x(t)$')
    plt.plot(tt, np.imag(z), label=r'Im$(z(t))$')
    plt.plot(tt, abs(z), label=r'$|z(t)|$')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'Signal level (V)')
    plt.title('Chirp with $f_c = %g$ Hz, $B = %g$ Hz, $T = %g$ s'%(fc, B, T))
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig('analytic.pdf')


    plt.show()



if __name__ == '__main__':
    main()
