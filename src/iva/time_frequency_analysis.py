# -*- coding: utf-8 -*-

import numpy as np

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming


class HammingShortTimeFFT:
    def __init__(self, fs: float, window_length: int, hop: int):
        self.sft = ShortTimeFFT(hamming(window_length), hop, fs)
        self.len_x = None

    def stft(self, x: np.ndarray) -> np.ndarray:
        self.len_x = x.shape[-1]
        return self.sft.stft(x)

    def istft(self, S: np.ndarray, len_x: int | None = None) -> np.ndarray:
        len_x = len_x or self.len_x
        if len_x is None:
            raise ValueError("`len_x`: length of the original signal must be provided.")
        return self.sft.istft(S, k1=len_x)
