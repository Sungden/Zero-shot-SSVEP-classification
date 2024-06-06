# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:14:51 2023

@author: Yang.D
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import os
import time
from scipy.io import loadmat
from config import config
import matplotlib.pyplot as plt
from model import universal_model 
from datetime import datetime
from FBCCA import fbcca,cca_reference
from function import GLST,Ref,generate_cca_references,filter_bank,generate_filterbank,itr,FB_stand,get_LST_transform
from CCA import FBTtCCA

# GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# cpu
cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(41)
LOOC = str(config.LOOC)+'_shot'
print('the shot is:',LOOC)
save_model_name = LOOC+'_TTCCA'
is_dataset = config.Dataset
num_harms = config.Nh

if is_dataset == 0:
    key_word = 'eeg'
    nCondition = 12
    rfs = 256  # sampling rate
    dataLength = 1114  # [-0.5 5.5s]
    nBlock = 15  # six blocks
    delay = 0.35  # 0.35 is better than 0.15
    latencyDelay = int(delay*rfs)  # 150ms delay 1114-1024
    list_freqs = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75,14.75]).T  # list of stimulus frequencies
    list_phase = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi  # list of stimulus phase
    name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    path1 = '/home/wc/Code/SSVEP_data/12JFPM/'  
    index_class = range(0, config.num_class)
    channels = [0, 1, 2, 3, 4, 5, 6, 7]  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(4*rfs)
    
elif is_dataset == 1:
    key_word = 'data'
    nCondition = 40
    rfs = 250  # sampling rate
    dataLength = 6 * rfs  # [-0.5 5.5s]
    nBlock = 6  # six blocks
    delay = 0.14 + 0.5  # visual latency being considered in the analysis [s]
    latencyDelay = int(delay * rfs)  # 140ms delay
    n_bands = 5  # number of sub-bands in filter bank analysis
    list_freqs = loadmat("/home/wc/Code/SSVEP_data/Bench/Freq_Phase.mat")['freqs'][0]
    #list_phase = loadmat("/data/yangdeng/data/Bench/Freq_Phase.mat")['phases'][0]
    list_phase=np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5])* np.pi 
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35']
    path1 = '/home/wc/Code/SSVEP_data/Bench/'  
    index_class = range(0, config.num_class)
    channels = [47, 53, 54, 55, 56, 57, 60, 61,62];  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(5*rfs)

elif is_dataset == 2:
    key_word = 'data'
    nCondition = 40
    rfs = 250  # sampling rate
    # dataLength = 3 * rfs  # [-0.5 2.5s]
    nBlock = 4  # six blocks
    delay = 0.13 + 0.5  # visual latency being considered in the analysis [s]
    latencyDelay = int(delay * rfs)  # 140ms delay
    list_freqs = loadmat("/home/wc/Code/SSVEP_data/BETA/Freqs_Beta.mat")['freqs'][0]
    list_phase = np.array(
        [1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
         1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1]) * np.pi
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
            'S47', 'S48', 'S49', 'S50', 'S51',
            'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
            'S67', 'S68', 'S69', 'S70']
    path1 = '/home/wc/Code/SSVEP_data/BETA/' 
    index_class = range(0, config.num_class)
    channels = [47, 53, 54, 55, 56, 57, 60, 61,62];  # Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
    signalLength=int(3*rfs)

path2 = '.mat'

PRmatrix_CCA = np.zeros(len(name))
PRmatrix_itr_CCA = np.zeros(len(name))
PRmatrix_TTCCA = np.zeros(len(name))
PRmatrix_itr_TTCCA = np.zeros(len(name))
PRmatrix_CCA_new= np.zeros(len(name))
PRmatrix_itr_CCA_new= np.zeros(len(name))
PRmatrix_TTCCA_new = np.zeros(len(name))
PRmatrix_itr_TTCCA_new= np.zeros(len(name))
PRmatrix_CCA_TTCCA= np.zeros(len(name))
PRmatrix_itr_CCA_TTCCA= np.zeros(len(name))
PRmatrix_CCA_TTCCA_new = np.zeros(len(name))
PRmatrix_itr_CCA_TTCCA_new= np.zeros(len(name))
PRmatrix_CCA_new_TTCCA= np.zeros(len(name))
PRmatrix_itr_CCA_new_TTCCA= np.zeros(len(name))
PRmatrix_CCA_new_TTCCA_new= np.zeros(len(name))
PRmatrix_itr_CCA_new_TTCCA_new =np.zeros(len(name))
PRmatrix_CCA_TTCCA_CCA_new= np.zeros(len(name))
PRmatrix_itr_CCA_TTCCA_CCA_new =np.zeros(len(name))
PRmatrix_CCA_TTCCA_CCA_new_TTCCA_new = np.zeros(len(name))
PRmatrix_itr_CCA_TTCCA_CCA_new_TTCCA_new= np.zeros(len(name))


num=0

for id_name in range(len(name)):
    if is_dataset == 0:
        name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    elif is_dataset == 1:
        name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
                'S32', 'S33', 'S34', 'S35']
    else:
        name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
                'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
                'S47', 'S48', 'S49', 'S50', 'S51',
                'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
                'S67', 'S68', 'S69', 'S70']
    name_test = [name[id_name]]
    name_train = name
    del name_train[id_name]
    name1 = name_train
    ##########################Training on source subjects###############################
    for i in range(len(name1)):
        path = path1 + name1[i] + path2
        mat = loadmat(path)
        data_raw = mat[key_word]
        if is_dataset == 0:
            data_raw = data_raw.transpose([1, 2, 0, 3])  # [C,T,num_class,nblock]
        elif is_dataset == 2:
            data_raw = data_raw[0][0][0]
            data_raw = data_raw.transpose([0, 1, 3, 2])  # [C,T,num_class,nblock]

        a = data_raw[channels, :, :, :]
        data_raw = a[:, latencyDelay:latencyDelay + config.T, index_class, :]
        data_raw = data_raw.transpose(2, 3, 0, 1)  # label*block*C*T
        data_raw_FB = filter_bank(data_raw) # label*block*3*C*T


        # LST transform
        LST_transform=np.zeros((data_raw.shape[0],data_raw.shape[1],2*config.Nh,data_raw.shape[3]))
        for cla in range(data_raw.shape[0]):
            P = GLST(data_raw[cla, :, :, :], rfs, num_harms, list_freqs[cla], list_phase[cla])
            for blo in range(data_raw.shape[1]):
                data_after = P @ data_raw[cla, blo, :, :]
                LST_transform[cla, blo :, :] = data_after

        if (i == 0):
            train_data = data_raw_FB
            train_data_raw = data_raw
            train_data_lst = LST_transform
        else:
            train_data = np.append(train_data, data_raw_FB, axis=1) # label*block*3*C*T
            train_data_raw = np.append(train_data_raw, data_raw, axis=1) # label*block*3*C*T
            train_data_lst = np.append(train_data_lst, LST_transform, axis=1) # label*block*8*T

    # data1c=filter_bank(train_datac)
    data1c=train_data
    data1c_raw=train_data_raw
    data1c_lst=train_data_lst
    print(data1c.shape)  # (40, 204,3, 9, 125)
    # print(data1c_lst.shape)  # (40, 204,8, 125)

    train_label=np.zeros((data1c.shape[0]*data1c.shape[1]))
    train_data_raw=np.zeros((data1c.shape[0]*data1c.shape[1],data1c.shape[3],data1c.shape[4]))
    train_data_lst=np.zeros((data1c.shape[0]*data1c.shape[1],2*config.Nh,data1c_lst.shape[3]))

    for j in range(config.num_class):
        train_data_raw[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = data1c_raw[j]
        train_label[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = np.ones(nBlock*len(name)) * j
        train_data_lst[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = data1c_lst[j]


    train_data_lst,Lst_weights,data_mean=get_LST_transform(train_data_raw, train_data_lst,train_label,list_freqs=config.list_freqs, fs=config.rfs, num_harms=4)



    ############################testing on target subject#####################################
    print('the second stage starting...')
    # print('the testing subject is:',name_test[0])
    path = path1 + name_test[0] + path2
    mat = loadmat(path)
    data_raw_test = mat[key_word]
    if is_dataset == 0:
        data_raw_test = data_raw_test.transpose([1, 2, 0, 3])
    elif is_dataset == 2:
        data_raw_test = data_raw_test[0][0][0]
        data_raw_test = data_raw_test.transpose([0, 1, 3, 2])  # [C,T,num_class,nblock] 

    b = data_raw_test[channels, :, :, :]
    data_raw_test = b[:, latencyDelay:latencyDelay + config.T, index_class, :]
    data_raw_test = data_raw_test.transpose(2, 3, 0, 1)  # label*block*C*T
    data1c_test = filter_bank(data_raw_test) #(num_class,block,3,channel,samples)



    ###########FBITCCA###########
    wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
    ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]
    filterbank = generate_filterbank(wp,ws,srate=rfs,order=15,rp=0.5)
    filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
    Yf = generate_cca_references(list_freqs, srate=rfs, T=config.sample_length,phases=list_phase,n_harmonics = 5)

    estimator_1 = FBTtCCA(filterbank=filterbank,n_components = 1,filterweights=np.array(filterweights), n_jobs=-1)
    estimator_1=estimator_1.fit(train_data_raw,train_label, Yf=Yf)
    estimator_2 = FBTtCCA(filterbank=filterbank,n_components = 1,filterweights=np.array(filterweights), n_jobs=-1)
    estimator_2=estimator_2.fit(train_data_raw,train_label, Yf=data_mean)

    t = time.time()
    data_test_raw=np.zeros((data_raw_test.shape[0]*data_raw_test.shape[1],data_raw_test.shape[2],data_raw_test.shape[3]))
    test_label=np.zeros((data1c_test.shape[0]*data1c_test.shape[1]))


    for j in range(config.num_class):
        data_test_raw[nBlock* j:nBlock * j + nBlock] = data_raw_test[j]
        test_label[nBlock* j:nBlock* j + nBlock] = np.ones(nBlock) * j


    p_labels_2,features_2 = estimator_1.predict(data_test_raw)
    p_labels_4,features_4 = estimator_2.predict(data_test_raw)
    p_labels_1,features_1 = fbcca(data_test_raw,y_ref = cca_reference(list_freqs, config.rfs, config.T, config.Nh))
    p_labels_3,features_3 = fbcca(data_test_raw,y_ref = data_mean)

    features_1=(features_1 - np.min(features_1, axis=1, keepdims=True)) / (np.max(features_1, axis=1, keepdims=True) - np.min(features_1, axis=1, keepdims=True))  
    features_2=(features_2 - np.min(features_2, axis=1, keepdims=True)) / (np.max(features_2, axis=1, keepdims=True) - np.min(features_2, axis=1, keepdims=True))  
    features_3=(features_3 - np.min(features_3, axis=1, keepdims=True)) / (np.max(features_3, axis=1, keepdims=True) - np.min(features_3, axis=1, keepdims=True))  
    features_4=(features_4 - np.min(features_4, axis=1, keepdims=True)) / (np.max(features_4, axis=1, keepdims=True) - np.min(features_4, axis=1, keepdims=True))  

    p_labels_new_1 = np.argmax(features_1+features_2, axis=-1)
    p_labels_new_2 = np.argmax(features_1+features_4, axis=-1)
    p_labels_new_3 = np.argmax(features_3+features_2, axis=-1)
    p_labels_new_4 = np.argmax(features_3+features_4, axis=-1)
    p_labels_new_5 = np.argmax(features_1+features_2+features_3, axis=-1)
    p_labels_new = np.argmax(features_1+features_2+features_3+features_4, axis=-1)
    print("testing time, the time cost is: {:.4f}s".format(time.time() - t)) 

    ncount_CCA= np.sum((p_labels_1 == test_label))
    ncount_TTCCA= np.sum((p_labels_2 == test_label))
    ncount_CCA_new= np.sum((p_labels_3 == test_label))
    ncount_TTCCA_new= np.sum((p_labels_4== test_label))
    ncount_CCA_TTCCA= np.sum((p_labels_new_1 == test_label))
    ncount_CCA_TTCCA_new= np.sum((p_labels_new_2 == test_label))
    ncount_CCA_new_TTCCA= np.sum((p_labels_new_3 == test_label))
    ncount_CCA_new_TTCCA_new= np.sum((p_labels_new_4 == test_label))
    ncount_CCA_TTCCA_CCA_new= np.sum((p_labels_new_5 == test_label))
    ncount_CCA_TTCCA_CCA_new_TTCCA_new= np.sum((p_labels_new == test_label))


    accuracy_CCA=ncount_CCA/(nBlock*config.num_class)
    itr_test_CCA = itr(config.num_class, accuracy_CCA, config.T / rfs + 0.5)

    accuracy_TTCCA=ncount_TTCCA/(nBlock*config.num_class)
    itr_test_TTCCA = itr(config.num_class, accuracy_TTCCA, config.T / rfs + 0.5)

    accuracy_CCA_new=ncount_CCA_new/(nBlock*config.num_class)
    itr_test_CCA_new = itr(config.num_class, accuracy_CCA_new, config.T / rfs + 0.5)

    accuracy_TTCCA_new=ncount_TTCCA_new/(nBlock*config.num_class)
    itr_test_TTCCA_new = itr(config.num_class, accuracy_TTCCA_new, config.T / rfs + 0.5)

    accuracy_CCA_TTCCA=ncount_CCA_TTCCA/(nBlock*config.num_class)
    itr_test_CCA_TTCCA = itr(config.num_class, accuracy_CCA_TTCCA, config.T / rfs + 0.5)

    accuracy_CCA_TTCCA_new=ncount_CCA_TTCCA_new/(nBlock*config.num_class)
    itr_test_CCA_TTCCA_new = itr(config.num_class, accuracy_CCA_TTCCA_new, config.T / rfs + 0.5)

    accuracy_CCA_new_TTCCA=ncount_CCA_new_TTCCA/(nBlock*config.num_class)
    itr_test_CCA_new_TTCCA = itr(config.num_class, accuracy_CCA_new_TTCCA, config.T / rfs + 0.5)

    accuracy_CCA_new_TTCCA_new=ncount_CCA_new_TTCCA_new/(nBlock*config.num_class)
    itr_test_CCA_new_TTCCA_new = itr(config.num_class, accuracy_CCA_new_TTCCA_new, config.T / rfs + 0.5)

    accuracy_CCA_TTCCA_CCA_new=ncount_CCA_TTCCA_CCA_new/(nBlock*config.num_class)
    itr_test_CCA_TTCCA_CCA_new = itr(config.num_class, accuracy_CCA_TTCCA_CCA_new, config.T / rfs + 0.5)

    accuracy_CCA_TTCCA_CCA_new_TTCCA_new=ncount_CCA_TTCCA_CCA_new_TTCCA_new/(nBlock*config.num_class)
    itr_test_CCA_TTCCA_CCA_new_TTCCA_new = itr(config.num_class, accuracy_CCA_TTCCA_CCA_new_TTCCA_new, config.T / rfs + 0.5)


    PRmatrix_CCA[id_name] = accuracy_CCA
    PRmatrix_itr_CCA[id_name] = itr_test_CCA

    PRmatrix_TTCCA[id_name] = accuracy_TTCCA
    PRmatrix_itr_TTCCA[id_name] = itr_test_TTCCA

    PRmatrix_CCA_new[id_name] = accuracy_CCA_new
    PRmatrix_itr_CCA_new[id_name] = itr_test_CCA_new

    PRmatrix_TTCCA_new[id_name] = accuracy_TTCCA_new
    PRmatrix_itr_TTCCA_new[id_name] = itr_test_TTCCA_new

    PRmatrix_CCA_TTCCA[id_name] = accuracy_CCA_TTCCA
    PRmatrix_itr_CCA_TTCCA[id_name] = itr_test_CCA_TTCCA

    PRmatrix_CCA_TTCCA_new[id_name] = accuracy_CCA_TTCCA_new
    PRmatrix_itr_CCA_TTCCA_new[id_name] = itr_test_CCA_TTCCA_new

    PRmatrix_CCA_new_TTCCA[id_name] = accuracy_CCA_new_TTCCA
    PRmatrix_itr_CCA_new_TTCCA[id_name] = itr_test_CCA_new_TTCCA

    PRmatrix_CCA_new_TTCCA_new[id_name] = accuracy_CCA_new_TTCCA_new
    PRmatrix_itr_CCA_new_TTCCA_new[id_name] = itr_test_CCA_new_TTCCA_new

    PRmatrix_CCA_TTCCA_CCA_new[id_name] = accuracy_CCA_TTCCA_CCA_new
    PRmatrix_itr_CCA_TTCCA_CCA_new[id_name] = itr_test_CCA_TTCCA_CCA_new

    PRmatrix_CCA_TTCCA_CCA_new_TTCCA_new[id_name] = accuracy_CCA_TTCCA_CCA_new_TTCCA_new
    PRmatrix_itr_CCA_TTCCA_CCA_new_TTCCA_new[id_name] = itr_test_CCA_TTCCA_CCA_new_TTCCA_new

    num += 1
    print('FBCCA accuracy is :',PRmatrix_CCA* 100)
    print(np.sum(PRmatrix_CCA * 100) /num)
    print('----------------------------')
    print('FBTTCCA accuracy is:',PRmatrix_TTCCA * 100)
    print(np.sum(PRmatrix_TTCCA * 100)  / num)
    print('----------------------------')
    print('FBCCA_new accuracy is :',PRmatrix_CCA_new * 100)
    print(np.sum(PRmatrix_CCA_new * 100) /num)
    print('----------------------------')
    print('FBTTCCA_new accuracy is :',PRmatrix_TTCCA_new * 100)
    print(np.sum(PRmatrix_TTCCA_new * 100) /num)
    print('----------------------------')
    print('FBCCA+FBTTCCA accuracy is:',PRmatrix_CCA_TTCCA * 100)
    print(np.sum(PRmatrix_CCA_TTCCA * 100)  / num)
    print('----------------------------')
    print('FBCCA+FBTTCCA_new accuracy is:',PRmatrix_CCA_TTCCA_new * 100)
    print(np.sum(PRmatrix_CCA_TTCCA_new * 100)  / num)
    print('----------------------------')
    print('FBCCA_new+FBTTCCA accuracy is:',PRmatrix_CCA_new_TTCCA * 100)
    print(np.sum(PRmatrix_CCA_new_TTCCA * 100)  / num)
    print('----------------------------')
    print('FBCCA_new+FBTTCCA_new accuracy is:',PRmatrix_CCA_new_TTCCA_new * 100)
    print(np.sum(PRmatrix_CCA_new_TTCCA_new * 100)  / num)
    print('----------------------------')
    print('CCA+TTCCA+CCA_new accuracy is:',PRmatrix_CCA_TTCCA_CCA_new * 100)
    print(np.sum(PRmatrix_CCA_TTCCA_CCA_new * 100)  / num)
    print('----------------------------')
    print('CCA+TTCCA+CCA_new+TTCCA_new accuracy is:',PRmatrix_CCA_TTCCA_CCA_new_TTCCA_new * 100)
    print(np.sum(PRmatrix_CCA_TTCCA_CCA_new_TTCCA_new * 100)  / num)
    print('----------------------------')


if is_dataset == 0:
    name = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
elif is_dataset == 1:
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35']
else:
    name = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
            'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
            'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
            'S47', 'S48', 'S49', 'S50', 'S51',
            'S52', 'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
            'S67', 'S68', 'S69', 'S70']

save_path = os.path.join(config.save_path, save_model_name, 'dataset_' + str(config.Dataset))
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_write_test_final = open(os.path.join(save_path, 'stimulus_' + str(config.T / rfs) + 's_' + str(LOOC) + "_"+str(datetime.now().strftime("%Y%m%d%H%M"))+"_test.txt"), "w")

##original
PRmatrix_CCA = PRmatrix_CCA * 100
acc = np.mean(PRmatrix_CCA)
var = np.var(PRmatrix_CCA)
std = np.sqrt(var)
itr_mean = np.mean(PRmatrix_itr_CCA)
itr_var = np.var(PRmatrix_itr_CCA)
itr_std = np.sqrt(itr_var)   


PRmatrix_TTCCA = PRmatrix_TTCCA * 100
acc_TTCCA = np.mean(PRmatrix_TTCCA)
var_TTCCA = np.var(PRmatrix_TTCCA)
std_TTCCA = np.sqrt(var_TTCCA)
itr_mean_TTCCA = np.mean(PRmatrix_itr_TTCCA)
itr_var_TTCCA = np.var(PRmatrix_itr_TTCCA)
itr_std_TTCCA = np.sqrt(itr_var_TTCCA) 


PRmatrix_FBCCA_new = PRmatrix_CCA_new * 100
acc_FBCCA_new = np.mean(PRmatrix_FBCCA_new)
var_FBCCA_new = np.var(PRmatrix_FBCCA_new)
std_FBCCA_new = np.sqrt(var_FBCCA_new)
itr_FBCCA_new= np.mean(PRmatrix_itr_CCA_new)
itr_var_FBCCA_new = np.var(PRmatrix_itr_CCA_new)
itr_std_FBCCA_new = np.sqrt(itr_var_FBCCA_new)


PRmatrix_TTCCA_new = PRmatrix_TTCCA_new * 100
acc_TTCCA_new = np.mean(PRmatrix_TTCCA_new)
var_TTCCA_new = np.var(PRmatrix_TTCCA_new)
std_TTCCA_new = np.sqrt(var_TTCCA_new)
itr_TTCCA_new = np.mean(PRmatrix_itr_TTCCA_new)
itr_var_TTCCA_new = np.var(PRmatrix_itr_TTCCA_new)
itr_std_TTCCA_new = np.sqrt(itr_var_TTCCA_new)   


PRmatrix_final_1 = PRmatrix_CCA_TTCCA * 100
acc_final_1 = np.mean(PRmatrix_final_1)
var_final_1 = np.var(PRmatrix_final_1)
std_final_1 = np.sqrt(var_final_1)
itr_final_1= np.mean(PRmatrix_itr_CCA_TTCCA)
itr_var_final_1 = np.var(PRmatrix_itr_CCA_TTCCA)
itr_std_final_1 = np.sqrt(itr_var_final_1)


####CNN+FBCCA_new
PRmatrix_final_3 = PRmatrix_CCA_TTCCA_new * 100
acc_final_3 = np.mean(PRmatrix_final_3)
var_final_3 = np.var(PRmatrix_final_3)
std_final_3 = np.sqrt(var_final_3)
itr_final_3= np.mean(PRmatrix_itr_CCA_TTCCA_new)
itr_var_final_3 = np.var(PRmatrix_itr_CCA_TTCCA_new)
itr_std_final_3 = np.sqrt(itr_var_final_3)

####CNN+FBTTCCA
PRmatrix_final_2 = PRmatrix_CCA_new_TTCCA * 100
acc_final_2 = np.mean(PRmatrix_final_2)
var_final_2 = np.var(PRmatrix_final_2)
std_final_2 = np.sqrt(var_final_2)
itr_final_2= np.mean(PRmatrix_itr_CCA_new_TTCCA)
itr_var_final_2 = np.var(PRmatrix_itr_CCA_new_TTCCA)
itr_std_final_2 = np.sqrt(itr_var_final_2)


####CNN+FBTTCCA+CCA_new
PRmatrix_final_4 = PRmatrix_CCA_new_TTCCA_new * 100
acc_final_4 = np.mean(PRmatrix_final_4)
var_final_4 = np.var(PRmatrix_final_4)
std_final_4 = np.sqrt(var_final_4)
itr_final_4= np.mean(PRmatrix_itr_CCA_new_TTCCA_new)
itr_var_final_4 = np.var(PRmatrix_itr_CCA_new_TTCCA_new)
itr_std_final_4 = np.sqrt(itr_var_final_4)

####CNN+FBTTCCA+CCA
PRmatrix_final_5 = PRmatrix_CCA_TTCCA_CCA_new * 100
acc_final_5 = np.mean(PRmatrix_final_5)
var_final_5 = np.var(PRmatrix_final_5)
std_final_5 = np.sqrt(var_final_5)
itr_final_5= np.mean(PRmatrix_itr_CCA_TTCCA_CCA_new)
itr_var_final_5 = np.var(PRmatrix_itr_CCA_TTCCA_CCA_new)
itr_std_final_5 = np.sqrt(itr_var_final_5)

####total
PRmatrix_final = PRmatrix_CCA_TTCCA_CCA_new_TTCCA_new * 100
acc_final = np.mean(PRmatrix_final)
var_final = np.var(PRmatrix_final)
std_final = np.sqrt(var_final)
itr_final= np.mean(PRmatrix_itr_CCA_TTCCA_CCA_new_TTCCA_new)
itr_var_final = np.var(PRmatrix_itr_CCA_TTCCA_CCA_new_TTCCA_new)
itr_std_final = np.sqrt(itr_var_final)

### original
log_write_test_final.write('the acc  of FBCCA is:' + "\n" + str(PRmatrix_CCA) + "\n")
log_write_test_final.write('the itr  of FBCCA is:' + "\n" + str(PRmatrix_itr_CCA) + "\n")
log_write_test_final.write('the mean acc of FBCCA is:' + str(acc) + "+-" + str(std / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA is:' + str(itr_mean) + "+-" + str(itr_std / np.sqrt(len(name))) + "\n")

log_write_test_final.write('the acc  of FBTTCCA is:' + "\n" + str(PRmatrix_TTCCA) + "\n")
log_write_test_final.write('the itr  of FBTTCCA is:' + "\n" + str(PRmatrix_itr_CCA) + "\n")
log_write_test_final.write('the mean acc of FBTTCCA is:' + str(acc_TTCCA) + "+-" + str(std_TTCCA / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBTTCCA is:' + str(itr_mean_TTCCA) + "+-" + str(itr_std_TTCCA / np.sqrt(len(name))) + "\n")

### gen_2
log_write_test_final.write('the acc  of FBCCA_new is:' + "\n" + str(PRmatrix_FBCCA_new) + "\n")
log_write_test_final.write('the itr  of FBCCA_new  is:' + "\n" + str(PRmatrix_itr_CCA_new) + "\n")
log_write_test_final.write('the mean acc of FBCCA_new is:' + str(acc_FBCCA_new) + "+-" + str(std_FBCCA_new / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA_new is:' + str(itr_FBCCA_new) + "+-" + str(itr_std_FBCCA_new / np.sqrt(len(name))) + "\n")

### CNN
log_write_test_final.write('the acc  of TTCCA_new is:' + "\n" + str(PRmatrix_TTCCA_new) + "\n")
log_write_test_final.write('the itr  of TTCCA_new is:' + "\n" + str(PRmatrix_itr_TTCCA_new) + "\n")
log_write_test_final.write('the mean acc of TTCCA_new is:' + str(acc_TTCCA_new) + "+-" + str(std_TTCCA_new / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of TTCCA_new is:' + str(itr_TTCCA_new) + "+-" + str(itr_std_TTCCA_new / np.sqrt(len(name))) + "\n")

### CNN+FBCCA
log_write_test_final.write('the acc  of FBCCA+FBTTCCA is:' + "\n" + str(PRmatrix_final_1) + "\n")
log_write_test_final.write('the itr  of FBCCA+FBTTCCA is:' + "\n" + str(PRmatrix_itr_CCA_TTCCA) + "\n")
log_write_test_final.write('the mean acc of FBCCA+FBTTCCA is:' + str(acc_final_1) + "+-" + str(std_final_1 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA+FBTTCCA is:' + str(itr_final_1) + "+-" + str(itr_std_final_1 / np.sqrt(len(name))) + "\n")


### CNN+FBCCA
log_write_test_final.write('the acc  of FBCCA+FBTTCCA_new is:' + "\n" + str(PRmatrix_final_3) + "\n")
log_write_test_final.write('the itr  of FBCCA+FBTTCCA_new is:' + "\n" + str(PRmatrix_itr_CCA_TTCCA_new) + "\n")
log_write_test_final.write('the mean acc of FBCCA+FBTTCCA_new is:' + str(acc_final_3) + "+-" + str(std_final_3 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA+FBTTCCA_new is:' + str(itr_final_3) + "+-" + str(itr_std_final_3 / np.sqrt(len(name))) + "\n")

### CNN+FBTTCCA
log_write_test_final.write('the acc  of FBCCA_new+FBTTCCA is:' + "\n" + str(PRmatrix_final_2) + "\n")
log_write_test_final.write('the itr  of FBCCA_new+FBTTCCA is:' + "\n" + str(PRmatrix_itr_CCA_new_TTCCA) + "\n")
log_write_test_final.write('the mean acc of FBCCA_new+FBTTCCA is:' + str(acc_final_2) + "+-" + str(std_final_2 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA_new+FBTTCCA is:' + str(itr_final_2) + "+-" + str(itr_std_final_2 / np.sqrt(len(name))) + "\n")


### CNN+FBTTCCA+CCA_new
log_write_test_final.write('the acc  of FBCCA_new+FBTTCCA_new is:' + "\n" + str(PRmatrix_final_4) + "\n")
log_write_test_final.write('the itr  of FBCCA_new+FBTTCCA_new is:' + "\n" + str(PRmatrix_itr_CCA_new_TTCCA_new) + "\n")
log_write_test_final.write('the mean acc of FBCCA_new+FBTTCCA_new is:' + str(acc_final_4) + "+-" + str(std_final_4 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA_new+FBTTCCA_new is:' + str(itr_final_4) + "+-" + str(itr_std_final_4 / np.sqrt(len(name))) + "\n")

### CNN+FBTTCCA+CCA
log_write_test_final.write('the acc  of FBCCA+FBTTCCA+FBCCA_new is:' + "\n" + str(PRmatrix_final_5) + "\n")
log_write_test_final.write('the itr  of FBCCA+FBTTCCA+FBCCA_new is:' + "\n" + str(PRmatrix_itr_CCA_TTCCA_CCA_new) + "\n")
log_write_test_final.write('the mean acc of FBCCA+FBTTCCA+FBCCA_new is:' + str(acc_final_5) + "+-" + str(std_final_5 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA+FBTTCCA+FBCCA_new is:' + str(itr_final_5) + "+-" + str(itr_std_final_5 / np.sqrt(len(name))) + "\n")


### final
log_write_test_final.write('the acc  of final is:' + "\n" + str(PRmatrix_final) + "\n")
log_write_test_final.write('the itr  of final is:' + "\n" + str(PRmatrix_itr_CCA_TTCCA_CCA_new_TTCCA_new) + "\n")
log_write_test_final.write('the mean acc of final is:' + str(acc_final) + "+-" + str(std_final / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of final is:' + str(itr_final) + "+-" + str(itr_std_final / np.sqrt(len(name))) + "\n")
# python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 main_gan.py
