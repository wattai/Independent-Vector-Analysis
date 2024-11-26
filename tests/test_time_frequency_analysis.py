# -*- coding: utf-8 -*-
import pytest
import numpy as np

from iva.time_frequency_analysis import HammingShortTimeFFT

SAMPLING_FREQUENCY = 16000
WINDOW_LENGTH = 512
HOP_LENGTH = 256
NUM_CHANNELS = 2


@pytest.fixture
def sft():
    return HammingShortTimeFFT(SAMPLING_FREQUENCY, WINDOW_LENGTH, HOP_LENGTH)


@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(SAMPLING_FREQUENCY),
        np.random.randn(NUM_CHANNELS, SAMPLING_FREQUENCY),
    ],
)
def test_sft(sft, x):
    out = sft.istft(sft.stft(x))
    assert out.shape == x.shape
    assert np.allclose(out, x)
