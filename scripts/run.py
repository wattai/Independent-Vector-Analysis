# -*- coding: utf-8 -*-

import numpy as np

from iva.independent_vector_analysis import _IndependentVectorAnalysis


def load_dummy_signals(
    time_length_sec=15,
    num_channels=2,
    fs=16000,
) -> np.ndarray:
    return np.random.randn(int(time_length_sec * fs), num_channels), fs


if __name__ == "__main__":
    # data, fs = sf.read("yuki_stereo_VM00_VF00_0750.wav")  # 2人の会話
    data, fs = load_dummy_signals()

    iva = _IndependentVectorAnalysis(
        num_components=2,
        fs=fs,
        num_iterations=5,
        fft_window_length=1024,
    )
    result = iva.fit_transform(data)
