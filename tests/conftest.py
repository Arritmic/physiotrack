"""Shared fixtures for the signals test suite.

Synthetic-signal generators live in ``tests/_synth.py`` so they can be imported
directly by test modules; this file only wires up fixtures.
"""
import os
import sys

import numpy as np
import pytest

# Make the sibling helper module importable regardless of pytest import mode.
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def rng():
    return np.random.RandomState(1234)
