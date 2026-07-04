from scipy import signal
import numpy as np


class LGI:
    """Local Group Invariance (LGI) blood-volume-pulse extractor.

    Recovers a blood-volume-pulse (BVP) signal from an RGB skin colour trace using
    a local group invariance projection: the first left-singular vector of the RGB
    trace is removed (projected out) and the second row of the residual is taken as
    the pulse. NumPy/CPU implementation of Pilz et al. (2018), "Local group
    invariance for heart rate estimation from face videos in the wild"
    (CVPR Workshops, pp. 1254-1262).

    Attributes:
        method_name (str): Method identifier, ``"LGI"``.
        projection (np.ndarray): Fixed ``(3, 3)`` projection matrix defined on the
            class (retained for reference; the pulse is derived from the SVD-based
            projection computed in :meth:`apply`).
        frameRate (float): Sampling rate (frames per second) of the input trace.

    Example:
        ```python
        from physiotrack.signals import LGI
        bvp = LGI(fps=30).apply(rgb_trace)   # rgb_trace: (3, N) rows R, G, B
        ```

    See Also:
        [`POS`][physiotrack.signals.POS], [`CHROM`][physiotrack.signals.CHROM],
        [`OMIT`][physiotrack.signals.OMIT]: other rPPG extraction methods.
    """
    method_name = 'LGI'
    projection = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])

    def __init__(self, fps):
        """Initialize the LGI extractor.

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

        U, _, _ = np.linalg.svd(signal)

        S = U[:, 0].reshape(1, -1)  # array 2D shape (1,3)
        P = np.identity(3) - np.matmul(S.T, S)

        Y = np.dot(P, signal)
        bvp = Y[1, :]

        return bvp