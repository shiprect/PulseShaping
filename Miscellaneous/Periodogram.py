from numpy import load, array, log10
from numpy.fft import fft, fftfreq, fftshift
from numpy.linalg import norm
from scipy.signal.windows import get_window



""" NOTE:
Name: periodogram
Purpose: Calculates the periodogram using the FFT of an input signal and a window function.
Parameters:
    xn -> Input signal.
    fft_size -> Size of the resultant FFT. If larger than the size of the input
                signal, the resultant FFT will be zero-padded.
    window -> The window function to perform on the FFT of the input signal.
              Window functions without extra arguments are passed as strings;
              those with extra arguments are passed as tuples of the form
              ('window_name', arg1, arg2, ...).
"""
def periodogram (Fs: int, xn: array, fft_size: int, window: str | tuple):
    # NOTE: The FFT is x[n] * e^-j(2pi/N)kn
    Xn = fft(xn, n=fft_size)
    wn = get_window(window, Xn.shape[0])
    return 1/(norm(wn)**2 * Fs) * abs(wn * Xn)**2