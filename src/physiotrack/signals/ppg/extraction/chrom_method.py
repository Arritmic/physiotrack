from scipy import signal
import numpy as np


class CHROM:
    """Chrominance-based (CHROM) blood-volume-pulse extractor.

    Recovers a blood-volume-pulse (BVP) signal from an RGB skin colour trace via
    two chrominance projections, ``X = 3R - 2G`` and ``Y = 1.5R + G - 1.5B``,
    combined as ``BVP = X - (sigma_X / sigma_Y) * Y`` to suppress the specular
    (motion) component. Described in Benezeth et al., "Remote heart rate
    variability for emotional state monitoring".

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
            np.ndarray: The 1-D blood-volume-pulse signal of length ``N``.
        """

        # calculation of new X and Y
        Xcomp = 3 * signal[0] - 2 * signal[1]
        Ycomp = (1.5 * signal[0]) + signal[1] - (1.5 * signal[2])

        # standard deviations
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)

        if sY != 0.0:
            alpha = sX / sY
        else:
            alpha = 1.0

        # -- rPPG signal
        bvp = Xcomp - alpha * Ycomp

        return bvp

