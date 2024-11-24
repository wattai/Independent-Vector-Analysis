# -*- coding: utf-8 -*-

import time

import numpy as np
import soundfile as sf

from iva.legacies import mystft


class IndependentVectorAnalysis:
    def __init__(self, data, num_sources):
        pass

    def fit(self):
        """
        Fit the model to the data using an iterative algorithm.

        This method estimates the independent components from the mixed data
        using a specified number of sources.

        Returns:
            self: The fitted model.
        """
        # Implementation of the fitting algorithm goes here
        return self

    def transform(self, data):
        pass


class _IndependentVectorAnalysis:
    def __init__(
        self,
        num_iterations: int = 10,
        fft_window_length: int = 128,
        num_components: int = 4,
        fs: float = 16000,
    ):
        """Independent Vector Analysis

        Args:
            num_iterations: The number of the iteration of IVA computaition.
            fft_window_length : Window length for FFT.
            num_components: The number of source signals.
            fs: Sampling frequency.
        """
        self.N = num_iterations
        self.fftLen = fft_window_length
        self.n_components = num_components
        self.fs = fs

        self.W = None  # Separation Matrix
        self.r = None  #
        self.spectrogram = None
        self.rebuild_spectrogram = None

    def _auxiva(self):
        # 独立ベクトル分析開始
        N_t = self.spectrogram.shape[0]
        N_omega = self.spectrogram.shape[1]
        K = self.spectrogram.shape[2]
        E = np.eye(K, dtype="complex")
        self.W = np.zeros([K, K, N_omega], dtype="complex")  # 分離行列初期化
        self.W[:, :, :] = E[:, :, None]
        self.r = np.zeros([K, N_t], dtype="complex")  # 時系列信号パワー初期化
        self.rebuild_spectrogram = np.zeros(
            [
                self.spectrogram.shape[0],
                self.spectrogram.shape[1],
                self.spectrogram.shape[2],
            ],
            dtype="complex",
        )
        z = 0 + 0j

        print(self.spectrogram.shape)

        # 反復回数N
        for i in range(self.N):
            for k in range(K):
                # 補助変数の更新1
                self.r[k, :] = np.squeeze(
                    np.sum(
                        (
                            np.abs(
                                self.spectrogram[:, :, :].T.transpose([1, 2, 0])
                                @ self.W[k, :, :][:, :, None]
                                .conj()
                                .transpose([1, 0, 2])
                            )
                        )
                        ** 2,
                        axis=0,
                    )
                )

                self.r[k, :] = np.sqrt(self.r[k, :])
                dr = np.gradient(self.r[k, :])
                G_R = self.r[k, :].copy()  # np.log(r[k, :]) # コントラスト関数指定
                fi = np.gradient(G_R, dr) / self.r[k, :]
                fi0 = 1000.0
                fi[fi0 < fi] = fi0

                # 補助変数の更新2
                V = (
                    (1 / N_t)
                    * (
                        ((fi * self.spectrogram.T).transpose([1, 0, 2]))
                        @ (self.spectrogram.conj().transpose([1, 0, 2]))
                    )
                ).transpose([1, 2, 0])
                # 分離行列の更新1(solve)
                self.W[k, :, :] = (
                    np.linalg.inv(
                        self.W.conj().transpose([2, 0, 1]) @ V.transpose([2, 0, 1])
                    )
                    @ E[k, :]
                ).T
                # 分離行列の更新2
                self.W[k, :, :] /= np.sqrt(
                    (
                        self.W[k, :, :][:, :, None].conj().transpose([1, 2, 0])
                        @ V.transpose([2, 0, 1])
                    )
                    @ self.W[k, :, :][:, :, None].transpose([1, 0, 2])
                ).squeeze()

            # 分離行列の正規化
            self.rebuild_spectrogram = (
                self.W.conj().transpose([2, 0, 1])
                @ self.spectrogram.transpose([1, 2, 0])
            ).transpose([2, 0, 1])
            z = np.sum(np.linalg.norm(self.rebuild_spectrogram, axis=1) ** 2)
            self.W[:] /= np.sqrt(z / (N_omega * N_t * K))

            print(str(i + 1) + "/" + str(self.N))

        # 信号源復元(分離処理)
        self.rebuild_spectrogram = (
            self.W.conj().transpose([2, 0, 1]) @ self.spectrogram.transpose([1, 2, 0])
        ).transpose([2, 0, 1])

        return self.rebuild_spectrogram

    def fit_transform(self, data):
        L, sigch = data.shape
        win = np.hamming(self.fftLen)  # ハミング窓
        step = (
            self.fftLen / 2
        ) / 2  # フレーム窓シフト幅(論文[一般で言われているシフト幅]のもう/2で合致？)
        if self.n_components is None:
            self.n_components = data.shape[1]
        elif self.n_components > data.shape[1]:
            self.n_components = data.shape[1]

        start = time.time()
        # Whitening ---------------------------------------------------
        # whited_data = whitening(data, self.n_components)
        whited_data = zca_whitening(data, self.n_components)
        ### --------------------------------------------------------------
        elapsed_time1 = time.time() - start
        print(whited_data.shape)

        sum_time = elapsed_time1
        elapsed_time2 = time.time() - start - sum_time

        # 時間領域 to 時間-周波数領域 --------------------------------------
        self.spectrogram = multi_stft(whited_data, win, step)  # STFT
        # -----------------------------------------------------------------
        sum_time += elapsed_time2
        elapsed_time3 = time.time() - start - sum_time

        ### AuxIVA --------------------------------------------------------
        self.rebuild_spectrogram = self._auxiva()
        ### --------------------------------------------------------------
        sum_time += elapsed_time3
        elapsed_time4 = time.time() - start - sum_time

        # 時間-周波数領域 to 時間領域 --------------------------------------
        result = multi_istft(self.rebuild_spectrogram, win, step)  # iSTFT
        result = result[
            len(result) - len(whited_data) :, :
        ]  # STFTで生じた余分な信号長のカット
        # result = multi_icwt(rebuild_spectrogram, omega0, sigma, fs) # iCWT(complex morlet)
        # -----------------------------------------------------------------
        sum_time += elapsed_time4
        elapsed_time5 = time.time() - start - sum_time

        print("PCA : {0:6.2f}".format(elapsed_time1) + "[sec]")
        print("FICA: {0:6.2f}".format(elapsed_time2) + "[sec]")
        print("STFT: {0:6.2f}".format(elapsed_time3) + "[sec]")
        print("IVA : {0:6.2f}".format(elapsed_time4) + "[sec]")
        print("iSTFT: {0:5.2f}".format(elapsed_time5) + "[sec]")

        print(np.sqrt(np.average(np.abs(data[:, :]) ** 2)))
        print(np.sqrt(np.average(np.abs(result[:, :]) ** 2)))

        print(np.linalg.norm(data[:, 0]))
        print(np.linalg.norm(result[:, 0]))

        # 振幅補正(RMSの比を基準に)
        # result[:, :] *= np.sqrt(np.average(np.abs(data[:, :])**2)) / np.sqrt(np.average(np.abs(result[:, :])**2))
        # 振幅補正(L2-norm の比を基準に)
        # result *= np.average(np.linalg.norm(data)) / np.linalg.norm(result)

        return result


def whitening(x, n_components):
    import numpy.linalg as LA

    x = x.copy()
    nData, nDim = np.shape(x)
    # 中心化centering
    x = x - np.mean(x, axis=0)
    # 相関行列
    C = np.dot(x.T, x) / nData
    # 共分散行列の固有値分解でE,Dを求める
    E, D, E_T = LA.svd(C)  # 元の
    # D, E = LA.eig(C)
    print(D)
    D = np.diag(D[:n_components] ** (-0.5))
    # 白色化行列V
    # V = np.dot(E, np.dot(D, E_T)) # 元の
    E = E[:, :n_components].copy()
    print(E.shape)
    print(E)
    V = D @ E.T.conj()  # PCA
    # 線形変換z
    z = x @ V.T
    return z


def zca_whitening(x, n_components):
    eps = 1e-6
    import numpy.linalg as LA

    x = x.copy()
    nData, nDim = np.shape(x)
    # 中心化centering
    x = x - np.mean(x, axis=0)
    # 相関行列
    C = np.dot(x.T, x) / nData
    # 共分散行列の固有値分解でE,Dを求める
    E, D, E_T = LA.svd(C)  # 元の
    # D, E = LA.eig(C)
    print(D)
    D = np.diag(1.0 / (np.sqrt(D[:n_components]) + eps))
    # D = np.diag(D[:n_components] ** (-0.5))
    # 白色化行列V
    # V = np.dot(E, np.dot(D, E_T)) # 元の
    E = E[:, :n_components].copy()
    print(E.shape)
    print(E)
    V = E @ D @ E.T.conj()  # ZCA
    # 線形変換z
    z = x @ V.T
    return z


def multi_stft(data, win, step):
    ### STFT ---------------------------------------------------------
    for i in range(data.shape[1]):
        if i == 0:
            buff = mystft.stft(data[:, i], win, step)
            spectrogram_ = np.empty(
                [buff.shape[0], buff.shape[1], data.shape[1]], dtype="complex"
            )
            spectrogram_[:, :, i] = buff
        if i > 0:
            spectrogram_[:, :, i] = mystft.stft(data[:, i], win, step)
    ### ---------------------------------------------------------------
    return spectrogram_


def multi_istft(rebuild_spectrogram, win, step):
    ### iSTFT ---------------------------------------------------------
    for i in range(rebuild_spectrogram.shape[2]):
        if i == 0:
            buff = mystft.istft(rebuild_spectrogram[:, :, i], win, step)
            resyn_data = np.empty([buff.shape[0], rebuild_spectrogram.shape[2]])
            resyn_data[:, i] = buff
        if i > 0:
            resyn_data[:, i] = mystft.istft(rebuild_spectrogram[:, :, i], win, step)
    ### ---------------------------------------------------------------
    return resyn_data


if __name__ == "__main__":
    data, samplerate = sf.read("yuki_stereo_VM00_VF00_0750.wav")  # 2人の会話
    iva = IndependentVectorAnalysis(N=5, fftLen=1024, n_components=2, fs=samplerate)
    result = iva.fit_transform(data)
