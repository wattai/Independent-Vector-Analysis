# -*- coding: utf-8 -*-


from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming


class HammingShortTimeFFT:
    def __init__(self, win_length, hop, fs):
        self.sft = ShortTimeFFT(hamming(win_length), hop, fs)

    def stft(self, x):
        return self.sft.stft(x)

    def istft(self, s):
        return self.sft.istft(s)
