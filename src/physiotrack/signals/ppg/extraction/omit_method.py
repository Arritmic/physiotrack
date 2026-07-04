from scipy import signal
import numpy as np


class OMIT:
    """Orthogonal Matrix Image Transformation (OMIT) blood-volume-pulse extractor.

    Recovers a blood-volume-pulse (BVP) signal from an RGB skin colour trace using
    a QR decomposition: the first orthonormal basis vector (``Q[:, 0]``) is
    projected out of the RGB trace and the second row of the residual is taken as
    the pulse. Described in Alvarez Casado, C. and Bordallo Lopez, M., "Face2PPG:
    Towards a reliable and unobtrusive blood volume pulse extraction from faces
    using RGB cameras".

    Attributes:
        method_name (str): Method identifier, ``"OMIT"``.
        frameRate (float): Sampling rate (frames per second) of the input trace.

    Example:
        ```python
        from physiotrack.signals import OMIT
        bvp = OMIT(fps=30).apply(rgb_trace)   # rgb_trace: (3, N) rows R, G, B
        ```

    See Also:
        [`POS`][physiotrack.signals.POS], [`CHROM`][physiotrack.signals.CHROM],
        [`LGI`][physiotrack.signals.LGI]: other rPPG extraction methods.
    """
    method_name = 'OMIT'

    def __init__(self, fps):
        """Initialize the OMIT extractor.

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
            np.ndarray: The 1-D blood-volume-pulse signal of length ``N`` (the
                second row of the projected trace).
        """
        Q, R = np.linalg.qr(signal)

        S = Q[:, 0].reshape(1, -1)  # array 2D shape (1,3)
        P = np.identity(3) - np.matmul(S.T, S)
        Y = np.dot(P, signal)
        bvp = Y[1, :]

        # bvp = -bvp

        return bvp
