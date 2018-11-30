# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:01:36 2016

@author: wattai
"""

import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros, hanning
from scipy.fftpack import fft, ifft
import scipy as sp
import soundfile as sf
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
import time
import concurrent.futures
from scipy.signal import fftconvolve, resample
import pandas as pd
import matplotlib.pyplot as plt
import numba

#from numba.decorators import jit, autojit
#from numba import guvectorize

def whitening(x, n_components):
    import numpy.linalg as LA
    x = x.copy()
    nData, nDim = np.shape(x)
    #中心化centering
    x = x - np.mean(x, axis=0)
    # 相関行列
    C = np.dot(x.T, x) / nData
    # 共分散行列の固有値分解でE,Dを求める
    E, D, E_T = LA.svd(C) # 元の
    #D, E = LA.eig(C)
    print(D)
    D = np.diag(D[:n_components] ** (-0.5))
    #白色化行列V
    #V = np.dot(E, np.dot(D, E_T)) # 元の
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
    #中心化centering
    x = x - np.mean(x, axis=0)
    # 相関行列
    C = np.dot(x.T, x) / nData
    # 共分散行列の固有値分解でE,Dを求める
    E, D, E_T = LA.svd(C) # 元の
    #D, E = LA.eig(C)
    print(D)
    D = np.diag(1.0 / (np.sqrt(D[:n_components]) + eps))
    #D = np.diag(D[:n_components] ** (-0.5))
    #白色化行列V
    #V = np.dot(E, np.dot(D, E_T)) # 元の
    E = E[:, :n_components].copy()
    print(E.shape)
    print(E)
    V = E @ D @ E.T.conj()  # ZCA
    # 線形変換z
    z = x @ V.T
    return z
    

class IndependentVectorAnalysis:
    
    def __init__(self, N=10, fftLen=128, n_components=4, fs=16000):
        
        self.N = N
        self.fftLen = fftLen
        self.n_components = n_components
        self.fs = fs
        self.W = None # Separation Matrix
        self.r = None # 
        self.spectrogram = None
        self.rebuild_spectrogram = None
        
    
    def _auxiva(self):

        # 独立ベクトル分析開始
        N_t = self.spectrogram.shape[0]
        N_omega = self.spectrogram.shape[1]
        K = self.spectrogram.shape[2]
        E = np.eye( K, dtype='complex' )
        self.W = np.zeros([K, K, N_omega], dtype='complex') # 分離行列初期化
        self.W[:, :, :] = E[:, :, None]
        self.r = np.zeros([ K, N_t ], dtype='complex') # 時系列信号パワー初期化
        #V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
        self.rebuild_spectrogram = np.zeros([ self.spectrogram.shape[0], self.spectrogram.shape[1], self.spectrogram.shape[2] ] , dtype='complex' )
        z = 0+0j
        
        print(self.spectrogram.shape)
        
        # 反復回数N
        for i in range( self.N ):
            for k in range( K ):
                # 補助変数の更新1
                self.r[k, :] = np.squeeze(np.sum((np.abs( self.spectrogram[:, :, :].T.transpose([1,2,0]) @ self.W[k, :, :][:, :, None].conj().transpose([1,0,2]) ))**2, axis=0))
                
                self.r[k, :] = np.sqrt( self.r[k, :] )
                dr = np.gradient(self.r[k, :])
                G_R = self.r[k, :].copy() #np.log(r[k, :]) # コントラスト関数指定
                fi = ( np.gradient(G_R, dr)/self.r[k, :] )
                fi0 = 1000.
                fi[fi0 < fi] = fi0
                
                # 補助変数の更新2
                V = ((1/N_t) * ((((fi*self.spectrogram.T).transpose([1,0,2]) ) @ (self.spectrogram.conj().transpose([1,0,2]) )))).transpose([1,2,0])
                # 分離行列の更新1(solve)            
                self.W[k, :, :] = (np.linalg.inv(self.W.conj().transpose([2,0,1]) @ V.transpose([2,0,1])) @ E[k, :]).T
                # 分離行列の更新2
                self.W[k, :, :] /= np.sqrt( (self.W[k, :, :][:, :, None].conj().transpose([1,2,0]) @ V.transpose([2,0,1])) @ self.W[k, :, :][:, :, None].transpose([1,0,2])  ).squeeze()
    
            # 分離行列の正規化      
            self.rebuild_spectrogram = ( self.W.conj().transpose([2,0,1]) @ self.spectrogram.transpose([1,2,0]) ).transpose([2,0,1])        
            z = np.sum(np.linalg.norm(self.rebuild_spectrogram, axis=1)**2)
            self.W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
            
            print( str(i+1) +"/" +str(self.N) )
        
        # 信号源復元(分離処理)
        self.rebuild_spectrogram = ( self.W.conj().transpose([2,0,1]) @ self.spectrogram.transpose([1,2,0]) ).transpose([2,0,1])
        
        return self.rebuild_spectrogram
        
    def fit_transform(self, data):

        L, sigch = data.shape
        win = sp.hamming(self.fftLen) # ハミング窓
        step = (self.fftLen/2) /2 # フレーム窓シフト幅(論文[一般で言われているシフト幅]のもう/2で合致？)
        if self.n_components is None:
            self.n_components = data.shape[1]
        elif self.n_components > data.shape[1]:
            self.n_components = data.shape[1]

        #from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA    
        start = time.time()
        # Whitening ---------------------------------------------------
        # pca = PCA(n_components=self.n_components, copy=True, whiten=True)
        # whited_data = pca.fit_transform( data[:, :] )
        #tSVD = TruncatedSVD(n_components=n_components, algorithm='arpack')
        #whited_data = tSVD.fit_transform( data )
        # whited_data = whitening(data, self.n_components) # 必ずこっちを使うこと
        whited_data = zca_whitening(data, self.n_components) # 必ずこっちを使うこと
        ### --------------------------------------------------------------
        elapsed_time1 = time.time() - start
        print(whited_data.shape)
        
        sum_time = elapsed_time1
        elapsed_time2 = time.time() - start - sum_time
      
        # CWT用パラメータ ---------------------------------------------------
        omega0 = 6
        sigma = 6
        N_scale = 256
        # ----------------------------------------------------------------
        # 時間領域 to 時間-周波数領域 --------------------------------------
        self.spectrogram = multi_stft(whited_data, win, step) # STFT
        # self.spectrogram = multi_cwt(whited_data, omega0, sigma, self.fs, N_scale) # CWT(complex morlet)
        # -----------------------------------------------------------------
        sum_time += elapsed_time2    
        elapsed_time3 = time.time() - start - sum_time
       
        ### AuxIVA --------------------------------------------------------
        self.rebuild_spectrogram = IndependentVectorAnalysis._auxiva(self)
        ### --------------------------------------------------------------
        sum_time += elapsed_time3
        elapsed_time4 = time.time() - start - sum_time
        
        # 時間-周波数領域 to 時間領域 --------------------------------------
        result = multi_istft(self.rebuild_spectrogram, win, step) # iSTFT
        result = result[len(result)-len(whited_data):, :] # STFTで生じた余分な信号長のカット
        #result = multi_icwt(rebuild_spectrogram, omega0, sigma, fs) # iCWT(complex morlet)
        # -----------------------------------------------------------------
        sum_time += elapsed_time4    
        elapsed_time5 = time.time() - start - sum_time
     
        print('PCA : {0:6.2f}'.format(elapsed_time1) + "[sec]")
        print('FICA: {0:6.2f}'.format(elapsed_time2) + "[sec]")
        print('STFT: {0:6.2f}'.format(elapsed_time3) + "[sec]")
        print('IVA : {0:6.2f}'.format(elapsed_time4) + "[sec]")
        print('iSTFT: {0:5.2f}'.format(elapsed_time5) + "[sec]")
    
        print(np.sqrt(np.average(np.abs(data[:, :])**2)))
        print(np.sqrt(np.average(np.abs(result[:, :])**2)))
        
        print(np.linalg.norm(data[:, 0]))
        print(np.linalg.norm(result[:, 0]))
        
        # 振幅補正(RMSの比を基準に)
        #result[:, :] *= np.sqrt(np.average(np.abs(data[:, :])**2)) / np.sqrt(np.average(np.abs(result[:, :])**2))
        # 振幅補正(L2-norm の比を基準に)
        #result *= np.average(np.linalg.norm(data)) / np.linalg.norm(result)
        
        return result
        
    def transform(self, data, W):

        self.W = W.copy() # 分離行列の取得
        
        L, sigch = data.shape
        win = sp.hamming(self.fftLen) # ハミング窓
        step = (self.fftLen/2) #/2 # フレーム窓シフト幅(論文[一般で言われているシフト幅]のもう/2しないとダメ?)
        if self.n_components is None: self.n_components = data.shape[1]
        elif self.n_components > data.shape[1]: self.n_components = data.shape[1]
        
        
        #from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA    
        start = time.time()
        # Whitening ---------------------------------------------------
        #pca = PCA(n_components=n_components, copy=True, whiten=True)
        #whited_data = pca.fit_transform( data[:, :] )
        #tSVD = TruncatedSVD(n_components=n_components, algorithm='arpack')
        #whited_data = tSVD.fit_transform( data )
        whited_data = whitening(data, self.n_components) # 必ずこっちを使うこと
        ### --------------------------------------------------------------
        elapsed_time1 = time.time() - start
        print(whited_data.shape)
        
        sum_time = elapsed_time1
        elapsed_time2 = time.time() - start - sum_time
      
        # CWT用パラメータ ---------------------------------------------------
        omega0 = 6
        sigma = 6
        N_scale = 256
        # ----------------------------------------------------------------
        # 時間領域 to 時間-周波数領域 --------------------------------------
        self.spectrogram = multi_stft(whited_data, win, step) # STFT
        #spectrogram = multi_cwt(whited_data, omega0, sigma, fs, N_scale) # CWT(complex morlet)
        # -----------------------------------------------------------------
        sum_time += elapsed_time2    
        elapsed_time3 = time.time() - start - sum_time
       
        ### 信号源復元(分離処理)
        self.rebuild_spectrogram = ( self.W.conj().transpose([2,0,1]) @ self.spectrogram.transpose([1,2,0]) ).transpose([2,0,1])
        ### --------------------------------------------------------------
        sum_time += elapsed_time3
        elapsed_time4 = time.time() - start - sum_time
        
        # 時間-周波数領域 to 時間領域 --------------------------------------
        result = multi_istft(self.rebuild_spectrogram, win, step) # iSTFT
        result = result[len(result)-len(whited_data):, :] # STFTで生じた余分な信号長のカット
        #result = multi_icwt(rebuild_spectrogram, omega0, sigma, fs) # iCWT(complex morlet)
        # -----------------------------------------------------------------
        sum_time += elapsed_time4    
        elapsed_time5 = time.time() - start - sum_time
     
        print('PCA : {0:6.2f}'.format(elapsed_time1) + "[sec]")
        print('FICA: {0:6.2f}'.format(elapsed_time2) + "[sec]")
        print('STFT: {0:6.2f}'.format(elapsed_time3) + "[sec]")
        print('IVA : {0:6.2f}'.format(elapsed_time4) + "[sec]")
        print('iSTFT: {0:5.2f}'.format(elapsed_time5) + "[sec]")
    
        print(np.sqrt(np.average(np.abs(data[:, :])**2)))
        print(np.sqrt(np.average(np.abs(result[:, :])**2)))
        
        print(np.linalg.norm(data[:, 0]))
        print(np.linalg.norm(result[:, 0]))
        
        # 振幅補正(RMSの比を基準に)
        #result[:, :] *= np.sqrt(np.average(np.abs(data[:, :])**2)) / np.sqrt(np.average(np.abs(result[:, :])**2))
        # 振幅補正(L2-norm の比を基準に)
        result *= np.average(np.linalg.norm(data)) / np.linalg.norm(result)
        
        return result        

"""
# AuxIVA1(多次元対応) -------------------------------------------------------
#@jit
def auxiva1(spectrogram, N):

    # 独立ベクトル分析開始
    N_t = spectrogram.shape[0]
    N_omega = spectrogram.shape[1]
    K = spectrogram.shape[2]
    E = np.identity( K, dtype='complex' )
    W = np.zeros([K, K, N_omega], dtype='complex')
    W[:, :, :] = E[:, :, None]
    r = np.zeros([ K, N_t ], dtype='complex')
    V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
    rebuild_spectrogram = np.zeros([ spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2] ] , dtype='complex' )
    z = 0+0j
        
    print(spectrogram.shape)
    
    # 反復回数N
    for i in range( N ):
        for k in range( K ):
            # 補助変数の更新1
            for omega in range( N_omega ):
                r[k, :] += (np.abs( W[k, :, omega].T.conj() @ ( spectrogram[:, omega, :].T )) )**2
            r[k, :] = np.sqrt( r[k, :] )
            dr = np.gradient(r[k, :])
            G_R = r[k, :].copy() #np.log(r[k, :]) # コントラスト関数指定
            fi = ( np.gradient(G_R, dr)/r[k, :] )
            fi0 = 1000.
            fi[fi0 < fi] = fi0

            for omega in range(N_omega):
                # 補助変数の更新2
                V2[:, :, k, omega] = (1/N_t) * (( fi *spectrogram[:, omega, :].T ) @ (spectrogram[:, omega, :].conj() )) # こっちのが速い
                # 分離行列の更新1(solve)
                W[k, :, omega] = np.linalg.solve( W[:, :, omega].conj() @ V2[:, :, k, omega], E[k, :]) # こっちのが速い（らしい）
                # 分離行列の更新2
                W[k, :, omega] = W[k, :, omega] / np.sqrt( W[k, :, omega].T.conj() @ V2[:, :, k, omega] @ W[k, :, omega] )

        # 分離行列の正規化
        for omega in range( N_omega ):
                rebuild_spectrogram[:, omega, :] = ( W[:, :, omega].conj() @ spectrogram[:, omega, :].T ).T
        for t in range(N_t):
            for k in range(K):
                z += np.linalg.norm(rebuild_spectrogram[t, :, k])**2
        #z = np.sum(r[:])**2
        W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
        
        print( str(i+1) +"/" +str(N) )
    
    # 信号源復元(分離処理)
    for omega in range( N_omega ):
        rebuild_spectrogram[:, omega, :] = ( W[:, :, omega].conj() @ spectrogram[:, omega, :].T ).T
    
    
    # Projection Back
    #n = 0
    #for omega in range( N_omega ):
    #    A = np.linalg.pinv(W[:, :, omega].conj())
    #    for t in range( N_t ):
    #        U = np.zeros([K], dtype='complex')
    #        #U[n] = rebuild_spectrogram[t, omega, n]
    #        U = rebuild_spectrogram[t, omega, :]
    #        v = (A - E) @ U
    #        rebuild_spectrogram[t, omega, :] = v / (2*(K**2))
    
    return rebuild_spectrogram
"""

def auxiva2(spectrogram, N):

    # 独立ベクトル分析開始
    N_t = spectrogram.shape[0]
    N_omega = spectrogram.shape[1]
    K = spectrogram.shape[2]
    E = np.identity( K, dtype='complex' )
    W = np.zeros([K, K, N_omega], dtype='complex')
    W[:, :, :] = E[:, :, None]
    r = np.zeros([ K, N_t ], dtype='complex')
    #V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
    rebuild_spectrogram = np.zeros([ spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2] ] , dtype='complex' )
    z = 0+0j
        
    print(spectrogram.shape)
    
    # 反復回数N
    for i in range( N ):
        for k in range( K ):
            # 補助変数の更新1
            r[k, :] = np.squeeze(np.sum((np.abs( spectrogram[:, :, :].T.transpose([1,2,0]) @ W[k, :, :][:, :, None].conj().transpose([1,0,2]) ))**2, axis=0))
            
            r[k, :] = np.sqrt( r[k, :] )
            dr = np.gradient(r[k, :])
            G_R = r[k, :].copy() #np.log(r[k, :]) # コントラスト関数指定
            fi = ( np.gradient(G_R, dr)/r[k, :] )
            fi0 = 1000.
            fi[fi0 < fi] = fi0
            
            # 補助変数の更新2
            V = ((1/N_t) * ( (((fi*spectrogram.T).transpose([1,0,2]) ) @ (spectrogram.conj().transpose([1,0,2]) )))).transpose([1,2,0])
            # 分離行列の更新1(solve)            
            W[k, :, :] = (np.linalg.inv(W.conj().transpose([2,0,1]) @ V.transpose([2,0,1])) @ E[k, :]).T
            # 分離行列の更新2
            W[k, :, :] /= np.sqrt( (W[k, :, :][:, :, None].conj().transpose([1,2,0]) @ V.transpose([2,0,1])) @ W[k, :, :][:, :, None].transpose([1,0,2])  ).squeeze()

        # 分離行列の正規化      
        rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])        
        z = np.sum(np.linalg.norm(rebuild_spectrogram, axis=1)**2)
        W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
        
        print( str(i+1) +"/" +str(N) )
    
    # 信号源復元(分離処理)
    rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])
    
    return rebuild_spectrogram
    
"""  
def auxiva3(spectrogram, N):

    # 独立ベクトル分析開始
    N_t = spectrogram.shape[0]
    N_omega = spectrogram.shape[1]
    K = spectrogram.shape[2]
    E = np.identity( K, dtype='complex' )
    W = np.zeros([N_omega, K, K], dtype='complex')
    W[:, :, :] = E[:, :, None].transpose([2,0,1])
    r = np.zeros([ K, N_t ], dtype='complex')
    #V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
    rebuild_spectrogram = np.zeros([ spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2] ] , dtype='complex' )
    z = 0+0j
        
    print(spectrogram.shape)
    spectrogram = spectrogram.transpose([1,0,2])
    
    # 反復回数N
    for i in range( N ):
        for k in range( K ):
            # 補助変数の更新1
            r[k, :] = np.squeeze(np.sum((np.abs( spectrogram @ W[:, k, :][:, :, None].conj() ))**2, axis=0))
            
            r[k, :] = np.sqrt( r[k, :] )
            dr = np.gradient(r[k, :])
            G_R = r[k, :].copy() #np.log(r[k, :]) # コントラスト関数指定
            fi = ( np.gradient(G_R, dr)/r[k, :] )
            fi0 = 1000.
            fi[fi0 < fi] = fi0

            # 補助変数の更新2
            V = ((1/N_t) * ( (((fi*spectrogram.transpose([0,2,1])) ) @ (spectrogram.conj() ))))
            # 分離行列の更新1(solve)
            W[:, k, :] = (np.linalg.inv(W.conj() @ V) @ E[k, :])
            
            # 分離行列の更新2
            w = W[:, k, :][:, :, None]
            W[:, k, :] = ( W[:, k, :].T / np.sqrt( w.conj().transpose([0,2,1]) @ V @ w ).squeeze() ).T

        # 分離行列の正規化      
        rebuild_spectrogram = ( W.conj() @ spectrogram.transpose([0,2,1]) ).transpose([2,0,1])        
        z = np.sum(np.linalg.norm(rebuild_spectrogram, axis=1)**2)
        W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
        
        print( str(i+1) +"/" +str(N) )
    
    # 信号源復元(分離処理)
    rebuild_spectrogram = ( W.conj() @ spectrogram.transpose([0,2,1]) ).transpose([2,0,1])
    
    return rebuild_spectrogram    
    
def auxiva4(spectrogram, N):

    # 独立ベクトル分析開始
    N_t = spectrogram.shape[0]
    N_omega = spectrogram.shape[1]
    K = spectrogram.shape[2]
    E = np.identity( K, dtype='complex' )
    W = np.zeros([K, K, N_omega], dtype='complex')
    W[:, :, :] = E[:, :, None]
    r = np.zeros([ K, N_t ], dtype='complex')
    #V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
    rebuild_spectrogram = np.zeros([ spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2] ] , dtype='complex' )
    z = 0+0j
        
    print(spectrogram.shape)
    
    # 反復回数N
    for i in range( N ):
        
        # 補助変数の更新1
        r[:, :] = np.einsum('ijk->kj', (np.abs( spectrogram.transpose([1,0,2]) @ W.conj().transpose([2,0,1]) ))**2)
        r[:, :] = np.sqrt( r[:, :] )
        dr = np.gradient(r[:, :], axis=1)
        G_R = r[:, :].copy() #np.log(r[k, :]) # コントラスト関数指定
        fi_ = ( np.gradient(G_R, dr, axis=1) / r[:, :] )
        fi0 = 1000.
        fi_[fi0 < fi_] = fi0
        
        # 補助変数の更新2
        V = ((1/N_t) * ( (((np.einsum('ijk,li->lijk', spectrogram, fi_).transpose([0,2,3,1]) ) @ (spectrogram.conj().transpose([1,0,2]) )))).transpose([0,2,3,1]))
           
        # 分離行列の更新1(solve)
        W = np.einsum( 'ijkk,ik->ikj', np.linalg.inv(W.conj().transpose([2,0,1]) @ V.transpose([0,3,1,2])), E[:, :] )        
           
        # 分離行列の更新2
        W /= np.sqrt( (W[:, :, :, None].conj().transpose([0,2,3,1]) @ V.transpose([0,3,1,2])) @ W[:, :, :, None].transpose([0,2,1,3])  ).squeeze()

        # 分離行列の正規化      
        rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])        
        z = np.sum(np.linalg.norm(rebuild_spectrogram, axis=1)**2)
        W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
        
        print( str(i+1) +"/" +str(N) )
    
    # 信号源復元(分離処理)
    rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])
    
    return rebuild_spectrogram
    
def auxiva5(spectrogram, N):

    from joblib import Parallel, delayed
    # 独立ベクトル分析開始
    N_t = spectrogram.shape[0]
    N_omega = spectrogram.shape[1]
    K = spectrogram.shape[2]
    E = np.identity( K, dtype='complex' )
    W = np.zeros([K, K, N_omega], dtype='complex')
    W[:, :, :] = E[:, :, None]
    r = np.zeros([ K, N_t ], dtype='complex')
    #V2 = np.zeros([ K, K, K, N_omega ], dtype='complex')
    rebuild_spectrogram = np.zeros([ spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2] ] , dtype='complex' )
    z = 0+0j
        
    print(spectrogram.shape)
    
    # 反復回数N
    for i in range( N ):

        W = Parallel(n_jobs=-1)( [delayed(aux_iva_sub_process)(k, r, spectrogram, W, N_t, E) for k in range(K)] )
        W = np.asarray(W, dtype='complex')
        # 分離行列の正規化      
        rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])        
        z = np.sum(np.linalg.norm(rebuild_spectrogram, axis=1)**2)
        W[:] /= (np.sqrt( z /(N_omega *N_t *K) ))
        
        print( str(i+1) +"/" +str(N) )
    
    # 信号源復元(分離処理)
    rebuild_spectrogram = ( W.conj().transpose([2,0,1]) @ spectrogram.transpose([1,2,0]) ).transpose([2,0,1])
    
    return rebuild_spectrogram

def aux_iva_sub_process(k, r, spectrogram, W, N_t, E):
    # 補助変数の更新1
    r[k, :] = np.squeeze(np.sum((np.abs( spectrogram[:, :, :].T.transpose([1,2,0]) @ W[k, :, :][:, :, None].conj().transpose([1,0,2]) ))**2, axis=0))
    
    r[k, :] = np.sqrt( r[k, :] )
    dr = np.gradient(r[k, :])
    G_R = r[k, :].copy() #np.log(r[k, :]) # コントラスト関数指定
    fi = ( np.gradient(G_R, dr)/r[k, :] )
    fi0 = 1000.
    fi[fi0 < fi] = fi0
    
    # 補助変数の更新2
    V = ((1/N_t) * ( (((fi*spectrogram.T).transpose([1,0,2]) ) @ (spectrogram.conj().transpose([1,0,2]) )))).transpose([1,2,0])
    # 分離行列の更新1(solve)            
    W[k, :, :] = (np.linalg.inv(W.conj().transpose([2,0,1]) @ V.transpose([2,0,1])) @ E[k, :]).T
    # 分離行列の更新2
    W[k, :, :] /= np.sqrt( (W[k, :, :][:, :, None].conj().transpose([1,2,0]) @ V.transpose([2,0,1])) @ W[k, :, :][:, :, None].transpose([1,0,2])  ).squeeze()

    return W[k, :, :]

"""

def multi_stft(data, win, step):
    import mystft
    ### STFT ---------------------------------------------------------
    for i in range(data.shape[1]):
        if i==0:
            buff = mystft.stft(data[:, i], win, step)
            spectrogram_ = np.empty([buff.shape[0], buff.shape[1], data.shape[1]], dtype='complex')
            spectrogram_[:, :, i] = buff
        if i>0:
            spectrogram_[:, :, i] = mystft.stft(data[:, i], win, step)
    ### ---------------------------------------------------------------
    return spectrogram_


def multi_istft(rebuild_spectrogram, win, step):
    import mystft
    ### iSTFT ---------------------------------------------------------
    for i in range( rebuild_spectrogram.shape[2] ):
        if i==0:
            buff = mystft.istft(rebuild_spectrogram[:, :, i], win, step)
            resyn_data = np.empty([buff.shape[0], rebuild_spectrogram.shape[2]])
            resyn_data[:, i] = buff
        if i>0:
            resyn_data[:, i] = mystft.istft(rebuild_spectrogram[:, :, i], win, step)
    ### ---------------------------------------------------------------
    return resyn_data
"""
def multi_cwt(data, omega0, sigma, fs, N_scale):
    import sys,os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../biological_signal_processing')
    import feature_extraction as fe
    ### CWT ---------------------------------------------------------
    for i in range(data.shape[1]):
        if i==0:
            buff = fe.cwt(data[:, i], omega0, sigma, fs, N_scale).T.copy()
            spectrogram = np.empty([buff.shape[0], buff.shape[1], data.shape[1]], dtype='complex')
            print(spectrogram.shape)
            spectrogram[:, :, i] = buff.copy()
        if i>0:
            spectrogram[:, :, i] = fe.cwt(data[:, i], omega0, sigma, fs, N_scale).T.copy()
    ### ---------------------------------------------------------------
    return spectrogram
    
def multi_icwt(rebuild_spectrogram, omega0, sigma, fs):
    import sys,os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../biological_signal_processing')
    import feature_extraction as fe
    ### iCWT ---------------------------------------------------------
    for i in range( rebuild_spectrogram.shape[2] ):
        if i==0:
            buff = fe.icwt(rebuild_spectrogram[:, :, i].T, omega0, sigma, fs).copy()
            resyn_data = np.empty([buff.shape[0], rebuild_spectrogram.shape[2]])
            resyn_data[:, i] = buff.copy()
        if i>0:
            resyn_data[:, i] = fe.icwt(rebuild_spectrogram[:, :, i].T, omega0, sigma, fs).copy()
    ### ---------------------------------------------------------------
    return resyn_data
"""
def IVA(data, N=10, fftLen=128, n_components=4, fs=16000):
    
    L, sigch = data.shape
    #N = 10 # 反復回数
    #fftLen = 8192/4 # とりあえず # フレーム窓長
    win = sp.hamming(fftLen) # ハミング窓
    step = (fftLen/2) #/2 # フレーム窓シフト幅(論文[一般で言われているシフト幅]のもう/2しないとダメ)
    #n_components = 4
    if n_components is None: n_components = data.shape[1]
    elif n_components > data.shape[1]: n_components = data.shape[1]
    
    
    #from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA    
    start = time.time()
    # Whitening ---------------------------------------------------
    #pca = PCA(n_components=n_components, copy=True, whiten=True)
    #whited_data = pca.fit_transform( data[:, :] )
    #tSVD = TruncatedSVD(n_components=n_components, algorithm='arpack')
    #whited_data = tSVD.fit_transform( data )
    whited_data = whitening(data, n_components) # 必ずこっちを使うこと
    ### --------------------------------------------------------------
    elapsed_time1 = time.time() - start
    print(whited_data.shape)
    
    sum_time = elapsed_time1
    elapsed_time2 = time.time() - start - sum_time
  
    # CWT用パラメータ ---------------------------------------------------
    omega0 = 6
    sigma = 6
    N_scale = 256
    # ----------------------------------------------------------------
    # 時間領域 to 時間-周波数領域 --------------------------------------
    spectrogram_ = multi_stft(whited_data, win, step) # STFT
    #spectrogram = multi_cwt(whited_data, omega0, sigma, fs, N_scale) # CWT(complex morlet)
    # -----------------------------------------------------------------
    sum_time += elapsed_time2    
    elapsed_time3 = time.time() - start - sum_time
   
    ### AuxIVA --------------------------------------------------------
    # Cython # python setup.py build_ext --inplace # でコンパイル
    #import pyximport; pyximport.install()#pyimport = True)
    #import iva_test
    #cProfile.run('iva_test.auxiva1( spectrogram, N )')
    #rebuild_spectrogram = iva_test.auxiva1( spectrogram, N )
    rebuild_spectrogram = auxiva2( spectrogram_, N )
    ### --------------------------------------------------------------
    sum_time += elapsed_time3
    elapsed_time4 = time.time() - start - sum_time
    
    # 時間-周波数領域 to 時間領域 --------------------------------------
    result = multi_istft(rebuild_spectrogram, win, step) # iSTFT
    # hanning窓の前半分をかけて入りを滑らかに
    #result[:int(fftLen/2), :] = (hanning(fftLen)[:int(fftLen/2)] * result[:int(fftLen/2), :].T).T
    result = result[len(result)-len(whited_data):, :] # STFTで生じた余分な信号長のカット
    #result = multi_icwt(rebuild_spectrogram, omega0, sigma, fs) # iCWT(complex morlet)
    # -----------------------------------------------------------------
    sum_time += elapsed_time4    
    elapsed_time5 = time.time() - start - sum_time
 
    print('PCA : {0:6.2f}'.format(elapsed_time1) + "[sec]")
    print('FICA: {0:6.2f}'.format(elapsed_time2) + "[sec]")
    print('STFT: {0:6.2f}'.format(elapsed_time3) + "[sec]")
    print('IVA : {0:6.2f}'.format(elapsed_time4) + "[sec]")
    print('iSTFT: {0:5.2f}'.format(elapsed_time5) + "[sec]")

    print(np.sqrt(np.average(np.abs(data[:, :])**2)))
    print(np.sqrt(np.average(np.abs(result[:, :])**2)))
    
    print(np.linalg.norm(data[:, 0]))
    print(np.linalg.norm(result[:, 0]))
    
    # 振幅補正(RMSの比を基準に)
    #result[:, :] *= np.sqrt(np.average(np.abs(data[:, :])**2)) / np.sqrt(np.average(np.abs(result[:, :])**2))
    # 振幅補正(L2-norm の比を基準に)
    result *= np.average(np.linalg.norm(data)) / np.linalg.norm(result)

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(data[:, 0])
    
    #plt.figure()    
    #plt.plot(result[:, 0])
    
 
    return result

if __name__ == "__main__":

    data, samplerate = sf.read("yuki_stereo_VM00_VF00_0750.wav")  # 2人の会話
    #data, samplerate = sf.read("townofdeath.wav")  # freeの曲

    if samplerate > 16000:
        print(samplerate)
        print("to")
        seconds = int(np.floor(len(data)/samplerate))
        size = samplerate *seconds
        data = data[:size, :].copy() # np.c_[data[:size, 0], data[:size, 1]]
        samplerate = 16000
        data = resample(data, samplerate*seconds)
        print(samplerate)
        print(seconds, "[sec]")

    # data = data[0*samplerate:30*samplerate, :]    
    
    """
    data1, samplerate = sf.read("dev2_ASY2016/dev2_mix4_asynchrec_realmix_ch12.wav")
    data2, samplerate = sf.read("dev2_ASY2016/dev2_mix4_asynchrec_realmix_ch34.wav")
    data3, samplerate = sf.read("dev2_ASY2016/dev2_mix4_asynchrec_realmix_ch56.wav")
    data4, samplerate = sf.read("dev2_ASY2016/dev2_mix4_asynchrec_realmix_ch78.wav")
    data =np.c_[data1, data2, data3, data4]
    """
    
    origin_data = data.copy() # 生データ保存
    
    
    # 関数版IVA --------------------------
    # result = IVA(data, N=20, fftLen=128*(2**1), n_components=2, fs=samplerate)
    # --------------------------------
    
    # クラス版IVA --------------------------
    iva = IndependentVectorAnalysis(N=5, fftLen=1024,
                                    n_components=2, fs=samplerate)
    result = iva.fit_transform(data)
    # --------------------------------
    print(result)
    
    W_im = np.abs(iva.W.reshape(128, -1))
    #print(W_im.shape)
    plt.pcolormesh(W_im)
    plt.colorbar()
    plt.show()
    xx = np.linalg.det(np.abs(iva.W.T))
    plt.plot(xx)
    plt.show()
    
    plt.plot(iva.r.T.real)
    plt.show()
    
    plt.plot(np.abs(iva.W.T.reshape(-1, 1)))
    plt.show()
    
    ss = []
    for i in range(iva.W.shape[2]):
        ss.append(np.sum(np.abs(iva.W[:, :, i])))
    ss = np.array(ss)
    plt.plot(ss)
    plt.show()
    
    for i in range(origin_data.shape[1]):
        sf.write('origin_file%d.wav' % i,
                 origin_data[:, i], samplerate, 'PCM_16')
    for i in range(result.shape[1]):
        sf.write('iva_file%d.wav' % i,
                 result[:, i], samplerate, 'PCM_16')

    sf.write('STEREO_ORIGIN.wav', origin_data, samplerate, 'PCM_16')
    sf.write('STEREO_IVA.wav', result, samplerate, 'PCM_16')

    """
    # 音声認識API ---------------------------------------------------------------
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../docomo_API')
    import docomo_stt
    speechtxt_origin = []
    speechtxt_iva = []
    for i in range(origin_data.shape[1]):
        path = ('./origin_file%d.wav' % i)
        speechtxt_origin.append(docomo_stt.stt(path))
    for i in range(result.shape[1]):
        path = ('./iva_file%d.wav' % i)
        speechtxt_iva.append(docomo_stt.stt(path))

    print("Result of Speech to TEXT")
    print("Origin File:")
    for i in range(origin_data.shape[1]):
        print(str(i) + ' 「' + speechtxt_origin[:][i] + '」')
    print("IVAed File:")
    for i in range(result.shape[1]):
        print(str(i) + ' 「' + speechtxt_iva[:][i] + '」')
    # --------------------------------------------------------------------------
    """
