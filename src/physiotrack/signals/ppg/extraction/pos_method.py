import numpy as np
from scipy import signal


class POS:
    """Plane-Orthogonal-to-Skin (POS) blood-volume-pulse extractor.

    Recovers a blood-volume-pulse (BVP) signal from an RGB skin colour trace by
    projecting the temporally-normalised RGB onto a plane orthogonal to the skin
    tone and overlap-adding the result across a sliding window. This is the POS
    method of Wang et al., "Algorithmic Principles of Remote PPG" (IEEE TBME,
    2017, https://ieeexplore.ieee.org/document/7565547).

    Attributes:
        method_name (str): Method identifier, ``"POS"``.
        projection (np.ndarray): Fixed ``(2, 3)`` projection matrix
            ``[[0, 1, -1], [-2, 1, 1]]`` applied to the normalised RGB.
        frameRate (float): Sampling rate (frames per second) used to size the
            sliding window.

    Example:
        ```python
        from physiotrack.signals import POS
        bvp = POS(fps=30).apply(rgb_trace)   # rgb_trace: (3, N) rows R, G, B
        ```

    See Also:
        [`CHROM`][physiotrack.signals.CHROM], [`LGI`][physiotrack.signals.LGI],
        [`OMIT`][physiotrack.signals.OMIT]: other rPPG extraction methods.
        [`HeartRateEstimator`][physiotrack.signals.HeartRateEstimator]: high-level
            estimator that wraps these methods.
    """

    method_name = 'POS'
    projection = np.array([[0, 1, -1], [-2, 1, 1]])

    def __init__(self, fps=30):
        """Initialize the POS extractor.

        Args:
            fps (float, optional): Frame rate of the RGB trace in Hz, used to set
                the sliding-window length (``1.6 s`` of samples). Defaults to
                ``30``.
        """
        self.frameRate = fps

    def apply(self, signal):
        """Extract the BVP signal from an RGB skin trace.

        Runs the POS algorithm over a sliding window of ``1.6 s`` (32 samples at
        20 fps) with overlap-add, per "Algorithm 1" of the paper.

        Args:
            signal (np.ndarray): RGB skin colour trace of shape ``(3, N)`` with
                rows ordered R, G, B and ``N`` time samples.

        Returns:
            np.ndarray: The 1-D blood-volume-pulse signal of length ``N``.
        """
        # Run the pos algorithm on the RGB color signal c with sliding window length wlen
        # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)

        wlen = int(1.6 * self.frameRate)

        # Initialize (1)
        h = np.zeros(signal.shape[1])
        for n in range(signal.shape[1]):
            # Start index of sliding window (4)
            m = n - wlen + 1
            if m >= 0:
                # Temporal normalization (5)
                cn = signal[:, m:(n + 1)]
                cn = np.dot(self.__get_normalization_matrix(cn), cn)
                # Projection (6)
                s = np.dot(self.projection, cn)

                std_s1 = np.std(s[1, :])
                if std_s1 == 0 or np.isnan(std_s1):   
                    continue
                
                # Tuning (7)
                hn = np.add(s[0, :], np.std(s[0, :]) / std_s1 * s[1, :])
                # print("Tino Info in POS class: {}, {}".format(n, s[1, :]))
                # Overlap-adding (8)
                h[m:(n + 1)] = np.add(h[m:(n + 1)], hn - np.mean(hn))
        return h

    def __get_normalization_matrix(self, x):
        # Compute a diagonal matrix n such that the mean of n*x is a vector of ones
        d = 0 if (len(x.shape) < 2) else 1
        m = np.mean(x, d)
        n = np.array([[1 / m[i] if i == j and m[i] else 0 for i in range(len(m))] for j in range(len(m))])
        return n