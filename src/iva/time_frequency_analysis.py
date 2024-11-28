# -*- coding: utf-8 -*-

import abc

import numpy as np

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming


class BaseShortTimeFFT(abc.ABC):
    @property
    def fs(self):
        return self._fs

    @property
    def window_length(self):
        return self._window_length

    @abc.abstractmethod
    def stft(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def istft(self, S: np.ndarray) -> np.ndarray:
        pass


class HammingShortTimeFFT(BaseShortTimeFFT):
    def __init__(self, fs: float, window_length: int, hop: int):
        self._fs = fs
        self._window_length = window_length
        self.sft = ShortTimeFFT(hamming(window_length), hop, fs)
        self.len_x = None

    def stft(self, x: np.ndarray) -> np.ndarray:
        self.len_x = x.shape[-1]
        return self.sft.stft(x)

    def istft(self, S: np.ndarray) -> np.ndarray:
        if self.len_x is None:
            raise ValueError("`len_x`: length of the original signal must be provided.")
        return self.sft.istft(S, k1=self.len_x)
