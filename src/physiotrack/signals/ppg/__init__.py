# -*- coding: utf-8 -*-
"""
Remote photoplethysmography (rPPG) utilities: blood-volume-pulse extraction
methods (POS/CHROM/LGI/OMIT).

Signal-filtering primitives (band-pass, detrend, normalisation) are not
duplicated here -- they live once in :mod:`physiotrack.signals.filters`.
"""
from .extraction import POS, CHROM, LGI, OMIT

__all__ = ["POS", "CHROM", "LGI", "OMIT"]
