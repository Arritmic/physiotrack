import numpy as np

from physiotrack.signals.filters import bandpass_filter

#: Band (Hz) used to compute the chrominance-mixing coefficient alpha. de Haan &
#: Jeanne band-pass X and Y before setting alpha = std(Xf)/std(Yf); this covers the
#: plausible pulse range (30-240 bpm). It is kept local (not imported from
#: ``constants``) to avoid a circular import with the method registry.
_ALPHA_BAND = (0.5, 4.0)


class CHROM:
    """Chrominance-based (CHROM) blood-volume-pulse extractor.

    Recovers a blood-volume-pulse (BVP) signal from an RGB skin colour trace via
    two chrominance projections. Following de Haan & Jeanne, "Robust pulse rate
    from chrominance-based rPPG" (*IEEE TBME* 60(10):2878-2886, 2013), each channel
    is first temporally mean-normalised (``Rn = R / mean(R)``, etc.) so the fixed
    projection coefficients operate on fractional reflectance changes; then
    ``X = 3Rn - 2Gn`` and ``Y = 1.5Rn + Gn - 1.5Bn`` are combined as
    ``BVP = X - (sigma_Xf / sigma_Yf) * Y`` where ``sigma_Xf``/``sigma_Yf`` are the
    standard deviations of the *band-passed* X and Y, so the specular/motion term is
    cancelled at pulse frequencies.

    Attributes:
        method_name (str): Method identifier, ``"CHROM"``.
        frameRate (float): Sampling rate (frames per second) of the input trace.

    Example:
        ```python
        from physiotrack.signals import CHROM
        bvp = CHROM(fps=30).apply(rgb_trace)   # rgb_trace: (3, N) rows R, G, B
        ```

    See Also:
        [`POS`][physiotrack.signals.POS], [`LGI`][physiotrack.signals.LGI],
        [`OMIT`][physiotrack.signals.OMIT]: other rPPG extraction methods.
    """
    method_name = 'CHROM'

    def __init__(self, fps):
        """Initialize the CHROM extractor.

        Args:
            fps (float): Frame rate of the RGB trace in Hz. Stored on the
                instance; not used directly by :meth:`apply`.
        """
        self.frameRate = fps

    def apply(self, signal):
        """Extract the BVP signal from an RGB skin trace.

        Args:
            signal (np.ndarray): RGB skin colour trace of shape ``(3, N)`` with
                rows ordered R, G, B and ``N`` time samples.

        Returns:
            np.ndarray: The 1-D blood-volume-pulse signal of length ``N``. The
                heart-rate band-pass filtering the rPPG pipeline applies downstream
                turns ``X - alpha*Y`` into the reference ``Xf - alpha*Yf``.
        """
        rgb = np.asarray(signal, dtype=float)

        # Per-channel temporal-mean normalization (de Haan & Jeanne 2013, Eq. 5-6):
        # convert each channel to fractional reflectance changes so the fixed
        # projection coefficients cancel the intensity/specular term. Guard against
        # a zero channel mean.
        means = np.mean(rgb, axis=1, keepdims=True)
        means[means == 0] = 1.0
        rgb_n = rgb / means

        # Chrominance projections on the normalized channels.
        Xcomp = 3.0 * rgb_n[0] - 2.0 * rgb_n[1]
        Ycomp = 1.5 * rgb_n[0] + rgb_n[1] - 1.5 * rgb_n[2]

        # alpha is set from the *band-passed* X/Y so the specular term is cancelled
        # in the pulse band (Eq. 6). Fall back to broadband std if the trace is too
        # short to filter.
        try:
            Xf = bandpass_filter(Xcomp, _ALPHA_BAND[0], _ALPHA_BAND[1], self.frameRate, order=3)
            Yf = bandpass_filter(Ycomp, _ALPHA_BAND[0], _ALPHA_BAND[1], self.frameRate, order=3)
        except Exception:
            Xf, Yf = Xcomp, Ycomp
        sX, sY = np.std(Xf), np.std(Yf)
        alpha = sX / sY if sY != 0.0 else 1.0

        # -- rPPG signal (left unfiltered; the pipeline band-passes it downstream).
        bvp = Xcomp - alpha * Ycomp
        return bvp

