import numpy as np
import scipy.sparse
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, detrend, firwin


def bandpass_filter(x, minHz, maxHz, fs, order=6):
    """Band-pass filter a signal with a Butterworth IIR design.

    Designs a Butterworth band-pass filter from the two cutoff frequencies
    (given directly in Hz, with ``fs`` supplied to ``scipy.signal.butter``)
    and applies it with a forward-only ``lfilter``. Because filtering is
    forward-only, the output has a frequency-dependent phase lag; use
    [`band_pass_filter`][physiotrack.signals.band_pass_filter] for the same
    design or the ``filtfilt``-based filters for zero-phase results.

    Args:
        x (np.ndarray): Input signal. A 1D array ``(N,)`` filters a single
            channel; a 2D array is filtered along its last axis.
        minHz (float): Lower cutoff frequency in Hz (band-pass low edge).
        maxHz (float): Upper cutoff frequency in Hz (band-pass high edge).
        fs (float): Sampling rate of ``x`` in Hz.
        order (int, optional): Butterworth filter order. Defaults to ``6``.

    Returns:
        np.ndarray: The filtered signal, same shape as ``x``.

    Example:
        ```python
        from physiotrack.signals.filters import bandpass_filter
        # Keep the 0.7-4 Hz (42-240 BPM) heart-rate band from a 30 fps signal
        filtered = bandpass_filter(rppg_trace, 0.7, 4.0, fs=30.0)
        ```

    See Also:
        [`band_pass_filter`][physiotrack.signals.band_pass_filter]: equivalent
            Butterworth band-pass taking the band as a pair.
    """

    # nyq = fs * 0.5
    # low = minHz/nyq
    # high = maxHz/nyq

    # print(low, high)
    # -- filter type
    # print('filtro=%f' % minHz)
    b, a = butter(order, Wn=[minHz, maxHz], fs=fs, btype='bandpass')

    # TODO verificare filtfilt o lfilter
    y = lfilter(b, a, x)
    # y = filtfilt(b, a, x)

    # w, h = freqz(b, a)

    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    # ax1.set_title('Digital filter frequency response')
    # ax1.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
    # ax1.set_ylabel('Amplitude [dB]', color='b')
    # plt.show()
    return y


def zero_mean_std_norm(x):
    """Standardize each row (channel) of a signal to zero mean and unit std.

    Normalizes along ``axis=1``, so every row of a ``(C, N)`` array is scaled
    independently to mean 0 and standard deviation 1. Intended for multi-channel
    signals such as the 3 RGB traces used in rPPG (1-3 channels).

    Args:
        x (np.ndarray): Signal of shape ``(C, N)`` with one channel per row.

    Returns:
        np.ndarray: Row-standardized signal, same shape ``(C, N)`` as ``x``.

    Example:
        ```python
        import numpy as np
        from physiotrack.signals.filters import zero_mean_std_norm
        rgb = np.random.rand(3, 300)   # 3 channels, 300 samples
        z = zero_mean_std_norm(rgb)    # each row now has mean 0, std 1
        ```

    See Also:
        [`zero_mean_std_norm_1ch`][physiotrack.signals.zero_mean_std_norm_1ch]:
            single-channel variant using a global mean/std.
    """
    # -- normalization along rows (1-3 channels)
    x = np.asarray(x, dtype=float)
    mx = x.mean(axis=1).reshape(-1, 1)
    sx = x.std(axis=1).reshape(-1, 1)
    # Guard constant channels (std == 0) to avoid divide-by-zero -> NaN propagation.
    sx = np.where(sx == 0, 1.0, sx)
    y = (x - mx) / sx
    return y


def zero_mean_std_norm_1ch(x):
    """Standardize a single-channel signal to zero mean and unit std.

    Uses the global mean and standard deviation of the whole array (unlike
    [`zero_mean_std_norm`][physiotrack.signals.zero_mean_std_norm], which
    normalizes per row), so it is appropriate for a single 1D trace.

    Args:
        x (np.ndarray): Input signal, typically 1D of shape ``(N,)``.

    Returns:
        np.ndarray: Standardized signal (mean 0, std 1) with the **same shape** as
            ``x``.

    Example:
        ```python
        import numpy as np
        from physiotrack.signals.filters import zero_mean_std_norm_1ch
        z = zero_mean_std_norm_1ch(np.random.rand(300))
        ```

    See Also:
        [`zero_mean_std_norm`][physiotrack.signals.zero_mean_std_norm]:
            multi-channel (per-row) variant.
    """
    # Global mean/std as scalars so the output keeps the input's shape (a 1-D
    # ``(N,)`` trace stays 1-D rather than being broadcast to ``(1, N)``).
    x = np.asarray(x, dtype=float)
    mx = x.mean()
    sx = x.std()
    if sx == 0:
        sx = 1.0
    y = (x - mx) / sx
    return y


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    """Design FIR band-pass filter coefficients with the window method.

    Builds a linear-phase FIR band-pass filter using ``scipy.signal.firwin``
    (``pass_zero=False``). This returns only the filter taps; apply them with a
    convolution routine such as ``scipy.signal.lfilter``.

    Args:
        ntaps (int): Number of filter taps (filter length). Larger values give
            a sharper transition band at the cost of more delay/computation.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling rate of the target signal in Hz.
        window (str, optional): Window function passed to ``firwin`` (e.g.
            ``"hamming"``, ``"hann"``, ``"blackman"``). Defaults to ``"hamming"``.

    Returns:
        np.ndarray: Array of ``ntaps`` FIR filter coefficients.

    Example:
        ```python
        from scipy.signal import lfilter
        from physiotrack.signals.filters import bandpass_firwin
        taps = bandpass_firwin(129, 0.7, 4.0, fs=30.0)
        filtered = lfilter(taps, 1.0, signal)
        ```
    """
    # Cutoffs are given directly in Hz; pass ``fs`` so ``firwin`` normalises them
    # internally. (The legacy ``nyq=`` keyword was removed in SciPy 1.12.)
    taps = firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False, window=window, scale=False)
    return taps


def band_pass_filter(signal, bandpass, fs, order=5):
    """Band-pass filter a signal with a Butterworth IIR design.

    Designs a Butterworth band-pass filter from the ``bandpass`` cutoff pair
    (in Hz, with ``fs`` passed to ``scipy.signal.butter``) and applies it with a
    forward-only ``lfilter``, so the output carries a phase lag. Equivalent to
    [`bandpass_filter`][physiotrack.signals.bandpass_filter] but takes the band
    as a single sequence rather than two separate arguments.

    Args:
        signal (np.ndarray): Input signal, filtered along its last axis.
        bandpass (Sequence[float]): Two-element ``[low, high]`` cutoff pair in Hz.
        fs (float): Sampling rate of ``signal`` in Hz.
        order (int, optional): Butterworth filter order. Defaults to ``5``.

    Returns:
        np.ndarray: The band-pass filtered signal, same shape as ``signal``.

    Example:
        ```python
        from physiotrack.signals.filters import band_pass_filter
        pulse = band_pass_filter(trace, [0.7, 4.0], fs=30.0)
        ```

    See Also:
        [`bandpass_filter`][physiotrack.signals.bandpass_filter]: same design
            with the band passed as ``minHz``/``maxHz`` arguments.
    """
    # Single implementation lives in ``bandpass_filter``; this is the two-argument
    # (band-as-pair) convenience wrapper so the two never diverge.
    low, high = bandpass
    return bandpass_filter(signal, low, high, fs, order=order)


def signaltonoise_dB(a, axis=0, ddof=0):
    """Compute the signal-to-noise ratio (SNR) in decibels.

    Estimates SNR as ``20 * log10(|mean / std|)`` along the given axis, i.e. the
    mean amplitude relative to the standard deviation, expressed in dB. Where the
    standard deviation is zero the ratio is treated as ``0`` to avoid division by
    zero (yielding ``-inf`` dB from ``log10(0)``).

    Args:
        a (array_like): Input signal; converted to an array internally.
        axis (int, optional): Axis along which to compute the SNR. Defaults to
            ``0``.
        ddof (int, optional): Delta degrees of freedom for the standard
            deviation. Defaults to ``0``.

    Returns:
        np.ndarray | float: SNR in dB, reduced along ``axis``. A scalar when the
            result collapses to a single value.

    Example:
        ```python
        from physiotrack.signals.filters import signaltonoise_dB
        snr = signaltonoise_dB(signal)
        ```
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))


def detrend_advanced(input_signal, detLambda=10, method='scipy'):
    """Detrend a signal, optionally with the Tarvainen smoothness-prior method.

    Removes slow trends from a 1D signal. With ``method="scipy"`` it applies
    ``scipy.signal.detrend`` (linear detrending). With ``method="Tarvainen"`` it
    uses the smoothness-prior approach of Tarvainen, Ranta-aho and Karjalainen
    ("An advanced detrending method with application to HRV analysis"), where the
    cutoff is controlled by ``detLambda``.

    Args:
        input_signal (np.ndarray): 1D signal of shape ``(N,)`` to detrend.
        detLambda (float, optional): Regularization parameter for the Tarvainen
            method; the internal smoothing lambda is ``N / detLambda``, so smaller
            ``detLambda`` removes lower frequencies. Only used when
            ``method="Tarvainen"``. Defaults to ``10``.
        method (str, optional): Detrending method, either ``"scipy"`` (linear) or
            ``"Tarvainen"`` (smoothness prior). Defaults to ``"scipy"``.

    Returns:
        np.ndarray: The detrended signal, same shape as ``input_signal``.

    Example:
        ```python
        from physiotrack.signals.filters import detrend_advanced
        clean = detrend_advanced(rppg_trace, detLambda=10, method="Tarvainen")
        ```

    Warning:
        The Tarvainen method builds and inverts an ``(N, N)`` matrix, so it is
        memory- and compute-heavy for long signals.
    """
    if method == 'Tarvainen':
        # Smoothness prior approach as in the paper appendix:
        # "An advanced detrending method with application to HRV analysis"
        # by Tarvainen, Ranta-aho and Karjaalainen
        t = input_signal.shape[0]
        l = t / detLambda  # lambda
        I = np.identity(t)
        D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t - 2, t)).toarray()
        detrended_signal = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(input_signal)
    else:
        detrended_signal = detrend(input_signal)

    return detrended_signal


def notch_filter(input_signal, notch_freq, sampling_rate):
    """Apply an IIR notch filter to remove a single frequency.

    Designs a second-order notch filter (``scipy.signal.iirnotch`` with a fixed
    quality factor ``Q = 30``) centered on ``notch_freq`` and applies it with a
    forward-only ``lfilter``. Useful for suppressing narrowband interference such
    as 50/60 Hz mains hum.

    Args:
        input_signal (np.ndarray): Input signal, filtered along its last axis.
        notch_freq (float): Frequency to remove, in Hz.
        sampling_rate (float): Sampling rate of ``input_signal`` in Hz.

    Returns:
        np.ndarray: The notch-filtered signal, same shape as ``input_signal``.

    Example:
        ```python
        from physiotrack.signals.filters import notch_filter
        clean = notch_filter(signal, notch_freq=50.0, sampling_rate=250.0)
        ```
    """
    Q = 30.0  # Quality factor
    b, a = signal.iirnotch(notch_freq, Q, sampling_rate)
    output_signal = signal.lfilter(b, a, input_signal)
    return output_signal


def highpass_filter(input_signal, cutoff_freq, sampling_rate, order=5):
    """Apply a zero-phase Butterworth high-pass filter.

    Designs a Butterworth high-pass filter (cutoff normalized by the Nyquist
    frequency) and applies it with ``filtfilt`` for zero phase distortion.
    Attenuates frequencies below ``cutoff_freq``.

    Args:
        input_signal (np.ndarray): Input signal, filtered along its last axis.
        cutoff_freq (float): High-pass cutoff frequency in Hz.
        sampling_rate (float): Sampling rate of ``input_signal`` in Hz.
        order (int, optional): Butterworth filter order. Defaults to ``5``.

    Returns:
        np.ndarray: The high-pass filtered signal, same shape as ``input_signal``.

    Example:
        ```python
        from physiotrack.signals.filters import highpass_filter
        detrended = highpass_filter(signal, cutoff_freq=0.5, sampling_rate=30.0)
        ```

    See Also:
        [`lowpass_filter`][physiotrack.signals.lowpass_filter]: complementary
            low-pass filter.
    """
    nyquist_freq = 0.5 * sampling_rate
    cutoff_norm = cutoff_freq / nyquist_freq
    b, a = butter(order, cutoff_norm, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, input_signal)
    return filtered_signal


def lowpass_filter(input_signal, cutoff_freq, sampling_rate, order=5):
    """Apply a zero-phase Butterworth low-pass filter.

    Designs a Butterworth low-pass filter (cutoff normalized by the Nyquist
    frequency) and applies it with ``filtfilt`` for zero phase distortion.
    Attenuates frequencies above ``cutoff_freq``.

    Args:
        input_signal (np.ndarray): Input signal, filtered along its last axis.
        cutoff_freq (float): Low-pass cutoff frequency in Hz.
        sampling_rate (float): Sampling rate of ``input_signal`` in Hz.
        order (int, optional): Butterworth filter order. Defaults to ``5``.

    Returns:
        np.ndarray: The low-pass filtered signal, same shape as ``input_signal``.

    Example:
        ```python
        from physiotrack.signals.filters import lowpass_filter
        smoothed = lowpass_filter(signal, cutoff_freq=4.0, sampling_rate=30.0)
        ```

    See Also:
        [`highpass_filter`][physiotrack.signals.highpass_filter]: complementary
            high-pass filter.
    """
    nyquist_freq = 0.5 * sampling_rate
    cutoff_norm = cutoff_freq / nyquist_freq
    b, a = butter(order, cutoff_norm, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, input_signal)
    return filtered_signal
