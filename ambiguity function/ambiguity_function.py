import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def ambgfun(x, *args, **kwargs):
    """
    Ambiguity function and cross ambiguity function.

    Parameters:
        x: ndarray
            Input signal vector.
        args: tuple
            Additional positional arguments:
                - y: ndarray (optional)
                  Second input signal for cross ambiguity function.
                - fs: float
                  Sampling frequency in Hz.
                - prf: float or list
                  Pulse repetition frequency (single value or list for both signals in cross mode).
        kwargs: dict
            Additional named arguments:
                - Cut: str
                  '2D', 'Delay', or 'Doppler' to specify the type of cut.
                - CutValue: float
                  Value at which the cut is applied in 'Delay' or 'Doppler' mode.

    Returns:
        Tuple containing ambiguity function and corresponding axes (time delay, Doppler).
        If no output is specified, it will plot the result.

    Examples:
        x = np.ones(10)
        fs = 10
        prf = 1
        ambgfun(x, fs, prf)
    """
    def parse_input(x, *args, **kwargs):
        cut = kwargs.get('Cut', '2D')
        cut_value = kwargs.get('CutValue', 0)

        if len(args) == 2:
            y = x
            fs, prf = args
            is_cross = False
        elif len(args) == 3:
            y, fs, prf = args
            is_cross = True
        else:
            raise ValueError("Invalid number of arguments.")

        if isinstance(prf, (list, tuple, np.ndarray)):
            x_prf, y_prf = prf
        else:
            x_prf = y_prf = prf

        return x, y, fs, x_prf, y_prf, cut, cut_value, is_cross

    def local_shift(x, tau, length):
        v = np.zeros(length, dtype=x.dtype)
        if tau >= 0:
            v[:length - tau] = x[tau:]
        else:
            v[-tau:] = x[:length + tau]
        return v

    # Parse input arguments
    x, y, fs, x_prf, y_prf, cut, cut_value, is_cross = parse_input(x, *args, **kwargs)

    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)

    seq_len = len(x_norm) + len(y_norm)
    tau = np.arange(-seq_len // 2 + 1, seq_len // 2)
    time_delay = tau / fs
    freq_bins = 2 ** int(np.ceil(np.log2(seq_len - 1)))
    doppler = np.fft.fftshift(np.fft.fftfreq(freq_bins, d=1 / fs))

    if cut == 'Delay':
        fft_x = np.fft.fft(x_norm, freq_bins)
        x_shift = np.fft.ifft(fft_x * np.exp(1j * 2 * np.pi * doppler * cut_value))
        fft_y = np.fft.fft(y_norm, freq_bins)
        y_original = np.fft.ifft(fft_y)
        af = np.abs(np.fft.ifftshift(np.fft.ifft(y_original * np.conj(x_shift), freq_bins)))
        return af, doppler
    elif cut == 'Doppler':
        time = np.arange(len(x_norm)) / fs
        fft_x = np.fft.fft(x_norm * np.exp(1j * 2 * np.pi * cut_value * time), seq_len)
        fft_y = np.fft.fft(y_norm, seq_len)
        af = np.abs(np.fft.fftshift(np.fft.ifft(fft_y * np.conj(fft_x))))
        return af, time_delay
    else:  # 2D Cut
        af = np.zeros((freq_bins, seq_len - 1))
        for m, t in enumerate(tau):
            shifted_x = local_shift(x_norm, t, len(x_norm))
            af[:, m] = np.abs(np.fft.ifftshift(np.fft.ifft(y_norm * np.conj(shifted_x), freq_bins)))
        af *= freq_bins
    return af, time_delay, doppler

