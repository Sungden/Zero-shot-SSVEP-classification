# -*- coding: utf-8 -*-
"""
Least-squares Transformation (LST).
See https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e.
"""
import numpy as np
from numpy import ndarray
from scipy.linalg import pinv
from config import config
from typing import Optional, List, Tuple, Union, cast
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from scipy import signal
from sklearn import preprocessing
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr

def lst_kernel(S: ndarray, T: ndarray):
    P = T @ S.T @ pinv(S @ S.T)
    return P

def GLST(data,fs,Nh,f,phase):
    """source signal estimation using LST [1]
    [1] https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
    Parameters
    ----------
    data : ndarray-like (block,n_channel_1, n_times)
        mean signal.
    mean_target : ndarray-like (n_channel_2, n_times)
        Reference signal.
    Returns
    -------
    data_after : ndarray-like (n_channel_2, n_times)
        Source signal.
    """
    if len(data.shape)!=3:
        data=np.expand_dims(data,axis=0)
    X_a = np.mean(data, axis=0)
    #  Generate reference signal Yf
    nChannel, nTime = X_a.shape
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((nTime, 2 * Nh))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[:, iNh * 2 + 1] = y_cos
    
    X = Yf.T    
    # Using the least squares method to solve aliasing matrix
    PT = lst_kernel(S=X_a, T=X) #(n_channel_2,n_channel_1)
   
    return PT


def get_LST_transform(eeg, eeg_lst,labels,list_freqs=config.list_freqs, list_phase=config.list_phase,fs=config.rfs, num_harms=4):
    unique_labels = np.unique(labels)
    data = np.zeros((len(unique_labels),eeg.shape[1],eeg.shape[2])) #num_class,C,T
    data_lst=np.zeros((len(unique_labels),eeg_lst.shape[1],eeg_lst.shape[2])) #num_class,2*Nh,T
    for label in unique_labels:
        mask = labels == label
        data_for_label = eeg[mask]
        data_for_label_lst = eeg_lst[mask]
        data[int(label)] = np.mean(data_for_label, axis=0)
        data_lst[int(label)] = np.mean(data_for_label_lst, axis=0)
    # LST transform
    LST_transform=np.zeros((eeg.shape[0],config.num_class,2*config.Nh,eeg.shape[2]))
    P_lst=np.zeros((data.shape[0],2*config.Nh,config.C))
    for cla in range(data.shape[0]):
        P= GLST(data[cla, :, :], fs, num_harms, list_freqs[cla], list_phase[cla])
        P_lst[cla]=P
        for idx in range(eeg.shape[0]):
            data_after = P @ eeg[idx, :, :]
            LST_transform[idx,cla, :, :] = data_after

    return LST_transform,P_lst,data_lst



def get_LST_transform_2(eeg, eeg_lst,labels,list_freqs=config.list_freqs, list_phase=config.list_phase,fs=config.rfs, num_harms=4):
    unique_labels = np.unique(labels)
    data = np.zeros((len(unique_labels),eeg.shape[1],eeg.shape[2])) #num_class,C,T
    data_lst=np.zeros((len(unique_labels),eeg_lst.shape[1],eeg_lst.shape[2])) #num_class,2*Nh,T
    for label in unique_labels:
        mask = labels == label
        data_for_label = eeg[mask]
        data_for_label_lst = eeg_lst[mask]
        data[int(label)] = np.mean(data_for_label, axis=0)
        data_lst[int(label)] = np.mean(data_for_label_lst, axis=0)
    # LST transform
    LST_transform=np.zeros((eeg.shape[0],config.num_class,2*config.Nh,eeg.shape[2]))
    P_lst=np.zeros((data.shape[0],2*config.Nh,config.C))
    LST_transform_2=np.zeros((eeg_lst.shape[0],config.num_class,2*config.Nh,eeg_lst.shape[2]))
    P_lst_2=np.zeros((data_lst.shape[0],2*config.Nh,2*config.Nh))
    for cla in range(data.shape[0]):
        P= GLST(data[cla, :, :], fs, num_harms, list_freqs[cla], list_phase[cla])
        P_lst[cla]=P
        P_2= GLST(data_lst[cla, :, :], fs, num_harms, list_freqs[cla], list_phase[cla])
        P_lst_2[cla]=P_2
        for idx in range(eeg.shape[0]):
            data_after = P @ eeg[idx, :, :]
            LST_transform[idx,cla, :, :] = data_after
            data_after_2 = P_2 @ eeg_lst[idx, :, :]
            LST_transform_2[idx,cla, :, :] = data_after_2

    return LST_transform_2,P_lst,P_lst_2,data_lst



def Ref(data,fs,Nh,f,phase):
    # data : ndarray-like (block,n_channel_1, n_times)
    #  Generate reference signal Yf
    nTime = data.shape[2]
    Ts = 1 / fs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((2 * Nh,nTime))

    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[iNh * 2,:] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
        Yf[iNh * 2 + 1,:] = y_cos
    return Yf

def Ref_total(fs_list,phase_list):
    # data : ndarray-like (block,n_channel_1, n_times)
    #  Generate reference signal Yf
    nTime = config.T
    Ts = 1 / config.rfs
    n = np.arange(nTime) * Ts
    Yf = np.zeros((2 * config.Nh,nTime))
    Y=np.zeros((config.num_class,2*config.Nh,config.T))
    for f,phase in zip(fs_list,phase_list):
        i=0
        for iNh in range(config.Nh):
            y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
            Yf[iNh * 2,:] = y_sin
            y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n+(iNh + 1)*phase)
            Yf[iNh * 2 + 1,:] = y_cos
        i+=1
        Y[i]=Yf
    return Y

def generate_cca_references(
    freqs: Union[ndarray, int, float],
    srate,
    T,
    phases: Optional[Union[ndarray, int, float]] = None,
    n_harmonics: int = 1,
):
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = np.array([freqs])
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = np.array([phases])
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T * srate))

    Yf = []
    for i in range(n_harmonics):
        Yf.append(
            np.stack(
                [
                    np.sin(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                    np.cos(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                ],
                axis=1,
            )
        )
    Yf = np.concatenate(Yf, axis=1)
    return Yf



# def filter_bank(eeg):
#     result = np.zeros((eeg.shape[0],eeg.shape[1],config.Nm,eeg.shape[2],eeg.shape[3]))# eeg shape:label*block*C*T
#     nyq = config.rfs / 2
#     passband = [5, 14, 22]
#     stopband = [3, 12, 20]
#     highcut_pass, highcut_stop = 90, 92
#     gpass, gstop, Rp = 3, 40, 0.5
#     for i in range(eeg.shape[0]):
#       for j in range(eeg.shape[1]):
#         for k in range(config.Nm):
#           Wp = [passband[k] / nyq, highcut_pass / nyq]
#           Ws = [stopband[k] / nyq, highcut_stop / nyq]
#           [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
#           [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
#           data = signal.filtfilt(B, A, eeg[i,j,:,:], padlen=3 * (max(len(B), len(A)) - 1)).copy()
#           result[i, j, k,:, :] = data
#     return result

## below implement is much better for UCSD dataset than upper implement(e.g. for 0.5s-long signal, the accuracy of DNN is 58.888 VS 41.50 ) , I don't know why
def filter_bank(eeg):
    result = np.zeros((eeg.shape[0],eeg.shape[1],config.Nm,eeg.shape[2],eeg.shape[3]))# eeg shape:label*block*C*T
    nyq = config.rfs / 2
    passband = [6, 14, 22]
    stopband = [4, 10, 16]
    highcut_pass, highcut_stop = 90, 100

    gpass, gstop, Rp = 3, 40, 0.5
    
    for i in range(eeg.shape[0]):
      for j in range(eeg.shape[1]):
        for k in range(config.Nm):
          Wp = [passband[k] / nyq, highcut_pass / nyq]
          Ws = [stopband[k] / nyq, highcut_stop / nyq]
          [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
          [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
          data = signal.filtfilt(B, A, eeg[i,j,:,:], padlen=3 * (max(len(B), len(A)) - 1)).copy()
          result[i, j, k,:, :] = data
    return result

def filter(eeg):
    result = np.zeros_like(eeg)# eeg shape:block*C*T
    nyq = config.rfs / 2
    passband = 6
    stopband = 4
    highcut_pass, highcut_stop = 90, 92
    gpass, gstop, Rp = 3, 40, 0.5

    Wp = [passband / nyq, highcut_pass / nyq]
    Ws = [stopband / nyq, highcut_stop / nyq]
    [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
    [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
    data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
    result = data
    return result

def generate_filterbank(
    passbands: List[Tuple[float, float]],
    stopbands: List[Tuple[float, float]],
    srate: int,
    order: Optional[int] = None,
    rp: float = 0.5,
):
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)

        filterbank.append(sos)
    return filterbank

def itr(n, p, t):
    if (p < 0 or 1 < p):
        raise ValueError('Accuracy need to be between 0 and 1.')
    elif (p < 1 / n):
        itr = 0
        # raise ValueError('ITR might be incorrect because accuracy < chance')
    elif (p == 1):
        itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p * np.log2(p) + (1 - p) *
               np.log2((1 - p) / (n - 1))) * 60 / t
    return itr


def FB_stand(Raw):
    b = Raw.shape[1]
    c = Raw.shape[0]
    Raw1 = np.zeros_like(Raw)
    for i_stand in range(b):
      for j_stand in range(c):
        Raw1[j_stand,i_stand] = preprocessing.scale(Raw[j_stand,i_stand],axis=1)
    return Raw1


def find_similarity(eeg_test,prediction_label, eeg_train,list_freqs=config.list_freqs, fs=config.rfs, num_harms=4):
    train_tmp=eeg_train[prediction_label] #subjects,C,T
    # print(train_tmp.shape,eeg_train.shape,'11111')#(9, 8, 128) (12, 9, 8, 128)
    y_ref = cca_reference(list_freqs, fs, train_tmp.shape[2], num_harms)
    cca = CCA(n_components=1) #initilize CCA
    r = np.zeros((train_tmp.shape[0]))
    for sub in range(train_tmp.shape[0]):
        train_C, _ = cca.fit_transform(np.squeeze(train_tmp[sub]).T, np.squeeze(y_ref[prediction_label]).T)
        test_C, _ = cca.fit_transform(np.squeeze(eeg_test).T, np.squeeze(y_ref[prediction_label]).T)
        r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(train_C))
        r[sub]=r_tmp

    top_5_indices =np.argsort(r)[-10:]
    return top_5_indices

def cca_reference(list_freqs, fs, num_smpls, num_harms=3):
    
    num_freqs = len(list_freqs)
    tidx = np.arange(1,num_smpls+1)/fs #time index
    
    y_ref = np.zeros((num_freqs, 2*num_harms, num_smpls))
    for freq_i in range(num_freqs):
        tmp = []
        for harm_i in range(1,num_harms+1):
            stim_freq = list_freqs[freq_i]  #in HZ
            # Sin and Cos
            tmp.extend([np.sin(2*np.pi*tidx*harm_i*stim_freq),
                       np.cos(2*np.pi*tidx*harm_i*stim_freq)])
        y_ref[freq_i] = tmp # 2*num_harms because include both sin and cos
    
    return y_ref