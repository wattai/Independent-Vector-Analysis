# -*- coding: utf-8 -*-

import pytest
import numpy as np

from iva.independent_vector_analysis import _IndependentVectorAnalysis


def load_dummy_signals(
    time_length_sec=15,
    num_channels=2,
    fs=16000,
) -> np.ndarray:
    return np.random.randn(int(time_length_sec * fs), num_channels), fs


@pytest.mark.parametrize(
    "num_components,num_iterations",
    [
        (2, 5),
        (3, 4),
        (4, 3),
    ],
)
def test_iva(
    num_components,
    num_iterations,
):
    input, fs = load_dummy_signals()

    iva = _IndependentVectorAnalysis(
        num_components=num_components,
        num_iterations=num_iterations,
        fs=fs,
        fft_window_length=1024,
    )
    out = iva.fit_transform(input)
    print(out)
