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
save_model_name = LOOC
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

PRmatrix = np.zeros(len(name))
PRmatrix_itr = np.zeros(len(name))
PRmatrix_TTCCA= np.zeros(len(name))
PRmatrix_itr_TTCCA = np.zeros(len(name))
PRmatrix_CCA = np.zeros(len(name))
PRmatrix_itr_CCA = np.zeros(len(name))
PRmatrix_CNN = np.zeros(len(name))
PRmatrix_itr_CNN = np.zeros(len(name))
PRmatrix_final_1 = np.zeros(len(name))
PRmatrix_itr_final_1 = np.zeros(len(name))
PRmatrix_final_2 = np.zeros(len(name))
PRmatrix_itr_final_2 = np.zeros(len(name))
PRmatrix_final_3 = np.zeros(len(name))
PRmatrix_itr_final_3 = np.zeros(len(name))
PRmatrix_final_4 = np.zeros(len(name))
PRmatrix_itr_final_4 = np.zeros(len(name))
PRmatrix_final_5 = np.zeros(len(name))
PRmatrix_itr_final_5 = np.zeros(len(name))
PRmatrix_final_6 = np.zeros(len(name))
PRmatrix_itr_final_6 = np.zeros(len(name))
PRmatrix_final = np.zeros(len(name))
PRmatrix_itr_final = np.zeros(len(name))

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


    train_data=np.zeros((data1c.shape[0]*data1c.shape[1],config.Nm,data1c.shape[3],data1c.shape[4]))
    train_label=np.zeros((data1c.shape[0]*data1c.shape[1]))
    reference=np.zeros((data1c.shape[0]*data1c.shape[1],2*config.Nh,data1c.shape[4]))
    train_data_lst=np.zeros((data1c.shape[0]*data1c.shape[1],2*config.Nh,data1c_lst.shape[3]))
    train_data_raw=np.zeros((data1c.shape[0]*data1c.shape[1],data1c.shape[3],data1c.shape[4]))

    for j in range(config.num_class):
        train_data[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = data1c[j]
        train_data_raw[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = data1c_raw[j]
        train_label[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = np.ones(nBlock*len(name)) * j
        train_data_lst[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = data1c_lst[j]
        reference[nBlock*len(name) * j:nBlock*len(name) * j + nBlock*len(name)] = Ref(data1c_raw[j],rfs,config.Nh,list_freqs[j],list_phase[j])
    
    train_data_lst,Lst_weights,data_mean=get_LST_transform(train_data_raw, train_data_lst,train_label,list_freqs=config.list_freqs, fs=config.rfs, num_harms=4)
    print('training input is:',train_data_raw.shape,train_data.shape,train_data_lst.shape,train_label.shape,data_mean.shape) #(8160, 3, 9, 125) (8160, 40, 8, 125) (8160,) (8160, 8, 125)

    train_data=FB_stand(train_data)
    train_data_lst=FB_stand(train_data_lst)
    # train_data_raw=FB_stand(train_data_raw)


    print('starting training Siamese -----------------')        
    t1=time.time()
    model =universal_model().train(train_data,train_data_lst,reference,train_label,config.n_epochs)
    print("finishing training Siamese,the time cost is: {:.4f}s".format(time.time() - t1))

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
    #LST
    data1c_test_new = np.zeros((data_raw_test.shape[0], data_raw_test.shape[1],config.num_class, 2 * config.Nh, data_raw_test.shape[3]))  # label*label*C*T
    for cla in range(config.num_class):
        P = Lst_weights[cla]
        for idx in range(data_raw_test.shape[0]):
            for blo in range(data_raw.shape[1]):
                data_after = P @ data_raw_test[idx,blo, :, :]
                data1c_test_new[idx, blo,cla, :, :] = data_after

 
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
    data_test=np.zeros((data1c_test.shape[0]*data1c_test.shape[1],data1c_test.shape[2],data1c_test.shape[3],data1c_test.shape[4]))
    data_test_raw=np.zeros((data_raw_test.shape[0]*data_raw_test.shape[1],data_raw_test.shape[2],data_raw_test.shape[3]))
    test_label=np.zeros((data1c_test.shape[0]*data1c_test.shape[1]))
    data_test_lst=np.zeros((data_raw_test.shape[0]*data_raw_test.shape[1],config.num_class,2*config.Nh,data_raw_test.shape[3]))

    for j in range(config.num_class):
        data_test[nBlock* j:nBlock * j + nBlock] = data1c_test[j]
        data_test_raw[nBlock* j:nBlock * j + nBlock] = data_raw_test[j]
        test_label[nBlock* j:nBlock* j + nBlock] = np.ones(nBlock) * j
        data_test_lst[nBlock* j:nBlock * j + nBlock] = data1c_test_new[j]

    data_test=FB_stand(data_test)
    data_test_lst=FB_stand(data_test_lst)

    prediction= universal_model().test(model,data_test,data_test_lst)
    p_labels_4,features_4 = estimator_1.predict(data_test_raw)
    p_labels_2,features_2 = estimator_2.predict(data_test_raw)
    p_labels_1,features_1 = fbcca(data_test_raw,y_ref = cca_reference(list_freqs, config.rfs, config.T, config.Nh))
    p_labels_3,features_3 = fbcca(data_test_raw,y_ref = data_mean)

    features_1=(features_1 - np.min(features_1, axis=1, keepdims=True)) / (np.max(features_1, axis=1, keepdims=True) - np.min(features_1, axis=1, keepdims=True))  
    features_2=(features_2 - np.min(features_2, axis=1, keepdims=True)) / (np.max(features_2, axis=1, keepdims=True) - np.min(features_2, axis=1, keepdims=True))  
    features_3=(features_3 - np.min(features_3, axis=1, keepdims=True)) / (np.max(features_3, axis=1, keepdims=True) - np.min(features_3, axis=1, keepdims=True))  
    features_4=(features_4 - np.min(features_4, axis=1, keepdims=True)) / (np.max(features_4, axis=1, keepdims=True) - np.min(features_4, axis=1, keepdims=True))  
    prediction=(prediction - np.min(prediction, axis=1, keepdims=True)) / (np.max(prediction, axis=1, keepdims=True) - np.min(prediction, axis=1, keepdims=True))  

    p_labels_new_0 = np.argmax(features_2+features_3, axis=-1)
    p_labels_new_1 = np.argmax(prediction+features_1, axis=-1)
    p_labels_new_2 = np.argmax(prediction+features_2, axis=-1)
    p_labels_new_3 = np.argmax(prediction+features_3, axis=-1)
    p_labels_new_6 = np.argmax(prediction+features_4, axis=-1)
    p_labels_new_4 = np.argmax(prediction+(features_3+features_2)/2, axis=-1)
    p_labels_new_5 = np.argmax(prediction+(features_1+features_2)/2, axis=-1)
    p_labels_new = np.argmax(prediction+(features_1+features_2+features_3)/3, axis=-1)
    print("testing time, the time cost is: {:.4f}s".format(time.time() - t)) 

    ncount= np.sum((p_labels_new_0 == test_label))
    ncount_TTCCA= np.sum((p_labels_2 == test_label))
    ncount_CCA= np.sum((p_labels_3 == test_label))
    ncount_CNN= np.sum((np.argmax(prediction,axis=-1) == test_label))
    ncount_total= np.sum((p_labels_new == test_label))
    ncount_total_1= np.sum((p_labels_new_1 == test_label))
    ncount_total_2= np.sum((p_labels_new_2 == test_label))
    ncount_total_3= np.sum((p_labels_new_3 == test_label))
    ncount_total_4= np.sum((p_labels_new_4 == test_label))
    ncount_total_5= np.sum((p_labels_new_5 == test_label))
    ncount_total_6= np.sum((p_labels_new_6 == test_label))


    accuracy=ncount/(nBlock*config.num_class)
    itr_test = itr(config.num_class, accuracy, config.T / rfs + 0.5)
    accuracy_CCA=ncount_CCA/(nBlock*config.num_class)
    itr_test_CCA = itr(config.num_class, accuracy_CCA, config.T / rfs + 0.5)
    accuracy_TTCCA=ncount_TTCCA/(nBlock*config.num_class)
    itr_test_TTCCA = itr(config.num_class, accuracy_TTCCA, config.T / rfs + 0.5)
    accuracy_CNN=ncount_CNN/(nBlock*config.num_class)
    itr_test_CNN = itr(config.num_class, accuracy_CNN, config.T / rfs + 0.5)
    accuracy_total_1=ncount_total_1/(nBlock*config.num_class)
    itr_test_total_1 = itr(config.num_class, accuracy_total_1, config.T / rfs + 0.5)
    accuracy_total_2=ncount_total_2/(nBlock*config.num_class)
    itr_test_total_2 = itr(config.num_class, accuracy_total_2, config.T / rfs + 0.5)
    accuracy_total_3=ncount_total_3/(nBlock*config.num_class)
    itr_test_total_3 = itr(config.num_class, accuracy_total_3, config.T / rfs + 0.5)
    accuracy_total_4=ncount_total_4/(nBlock*config.num_class)
    itr_test_total_4 = itr(config.num_class, accuracy_total_4, config.T / rfs + 0.5)
    accuracy_total_5=ncount_total_5/(nBlock*config.num_class)
    itr_test_total_5 = itr(config.num_class, accuracy_total_5, config.T / rfs + 0.5)
    accuracy_total_6=ncount_total_6/(nBlock*config.num_class)
    itr_test_total_6 = itr(config.num_class, accuracy_total_6, config.T / rfs + 0.5)
    accuracy_total=ncount_total/(nBlock*config.num_class)
    itr_test_total = itr(config.num_class, accuracy_total, config.T / rfs + 0.5)

    # print(name_test[0], " FBCCA Test set results:", "Accuracy= {:.4f}".format(accuracy))
    # print(name_test[0], " FBTTCCA Test set results:", "Accuracy= {:.4f}".format(accuracy_TTCCA))
    # print(name_test[0], " FBCCA_new Test set results:", "Accuracy= {:.4f}".format(accuracy_CCA))
    # print(name_test[0], " CNN Test set results:", "Accuracy= {:.4f}".format(accuracy_CNN))
    # print(name_test[0], " CNN+FBCCA Test set results:", "Accuracy= {:.4f}".format(accuracy_total_1))
    # print(name_test[0], " CNN+FBTTCCA Test set results:", "Accuracy= {:.4f}".format(accuracy_total_2))
    # print(name_test[0], " CNN+FBCCA_new Test set results:", "Accuracy= {:.4f}".format(accuracy_total_3))
    # print(name_test[0], " CNN+FBCCA_new+TTCCA Test set results:", "Accuracy= {:.4f}".format(accuracy_total_4))
    # print(name_test[0], " CNN+FBCCA+TTCCA Test set results:", "Accuracy= {:.4f}".format(accuracy_total_5))
    # print(name_test[0], " Final Test set results:", "Accuracy= {:.4f}".format(accuracy_total))
    


    PRmatrix[id_name] = accuracy
    PRmatrix_itr[id_name] = itr_test

    PRmatrix_CCA[id_name] = accuracy_CCA
    PRmatrix_itr_CCA[id_name] = itr_test_CCA

    PRmatrix_TTCCA[id_name] = accuracy_TTCCA
    PRmatrix_itr_TTCCA[id_name] = itr_test_TTCCA

    PRmatrix_CNN[id_name] = accuracy_CNN
    PRmatrix_itr_CNN[id_name] = itr_test_CNN

    PRmatrix_final_1[id_name] = accuracy_total_1
    PRmatrix_itr_final_1[id_name] = itr_test_total_1

    PRmatrix_final_2[id_name] = accuracy_total_2
    PRmatrix_itr_final_2[id_name] = itr_test_total_2

    PRmatrix_final_3[id_name] = accuracy_total_3
    PRmatrix_itr_final_3[id_name] = itr_test_total_3

    PRmatrix_final_4[id_name] = accuracy_total_4
    PRmatrix_itr_final_4[id_name] = itr_test_total_4

    PRmatrix_final_5[id_name] = accuracy_total_5
    PRmatrix_itr_final_5[id_name] = itr_test_total_5

    PRmatrix_final_6[id_name] = accuracy_total_6
    PRmatrix_itr_final_6[id_name] = itr_test_total_6


    PRmatrix_final[id_name] = accuracy_total
    PRmatrix_itr_final[id_name] = itr_test_total

    num += 1
    print('FBTTCCA_new+CCA_new accuracy is :',PRmatrix* 100)
    print(np.sum(PRmatrix * 100) /num)
    print('----------------------------')
    print('FBTTCCA accuracy is:',PRmatrix_TTCCA * 100)
    print(np.sum(PRmatrix_TTCCA * 100)  / num)
    print('----------------------------')
    print('CNN accuracy is :',PRmatrix_CNN * 100)
    print(np.sum(PRmatrix_CNN * 100) /num)
    print('----------------------------')
    print('CCA_new accuracy is :',PRmatrix_CCA * 100)
    print(np.sum(PRmatrix_CCA * 100) /num)
    print('----------------------------')
    print('CNN+FBCCA accuracy is:',PRmatrix_final_1 * 100)
    print(np.sum(PRmatrix_final_1 * 100)  / num)
    print('----------------------------')
    print('CNN+FBTTCCA_new accuracy is:',PRmatrix_final_2 * 100)
    print(np.sum(PRmatrix_final_2 * 100)  / num)
    print('----------------------------')
    print('CNN+FBCCA_new accuracy is:',PRmatrix_final_3 * 100)
    print(np.sum(PRmatrix_final_3 * 100)  / num)
    print('----------------------------')
    print('CNN+FBTTCCA_new+FBCCA_new accuracy is:',PRmatrix_final_4 * 100)
    print(np.sum(PRmatrix_final_4 * 100)  / num)
    print('----------------------------')
    print('CNN+FBTTCCA_new+FBCCA accuracy is:',PRmatrix_final_5 * 100)
    print(np.sum(PRmatrix_final_5 * 100)  / num)
    print('----------------------------')
    print('CNN+FBTTCCA accuracy is:',PRmatrix_final_6 * 100)
    print(np.sum(PRmatrix_final_6 * 100)  / num)
    print('----------------------------')
    print('Final accuracy is:',PRmatrix_final * 100)
    print(np.sum(PRmatrix_final * 100)  / num)
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
PRmatrix = PRmatrix * 100
acc = np.mean(PRmatrix)
var = np.var(PRmatrix)
std = np.sqrt(var)
itr_mean = np.mean(PRmatrix_itr)
itr_var = np.var(PRmatrix_itr)
itr_std = np.sqrt(itr_var)   


PRmatrix_CCA = PRmatrix_CCA * 100
acc_CCA = np.mean(PRmatrix_CCA)
var_CCA = np.var(PRmatrix_CCA)
std_CCA = np.sqrt(var_CCA)
itr_mean_CCA = np.mean(PRmatrix_itr_CCA)
itr_var_CCA = np.var(PRmatrix_itr_CCA)
itr_std_CCA = np.sqrt(itr_var_CCA) 


####gen_2
PRmatrix_TTCCA = PRmatrix_TTCCA * 100
acc_TTCCA = np.mean(PRmatrix_TTCCA)
var_TTCCA = np.var(PRmatrix_TTCCA)
std_TTCCA = np.sqrt(var_TTCCA)
itr_TTCCA= np.mean(PRmatrix_itr_TTCCA)
itr_var_TTCCA = np.var(PRmatrix_itr_TTCCA)
itr_std_TTCCA = np.sqrt(itr_var_TTCCA)

##CNN
PRmatrix_CNN = PRmatrix_CNN * 100
acc_CNN = np.mean(PRmatrix_CNN)
var_CNN = np.var(PRmatrix_CNN)
std_CNN = np.sqrt(var_CNN)
itr_CNN = np.mean(PRmatrix_itr_CNN)
itr_var_CNN = np.var(PRmatrix_itr_CNN)
itr_std_CNN = np.sqrt(itr_var_CNN)   

####CNN+FBCCA
PRmatrix_final_1 = PRmatrix_final_1 * 100
acc_final_1 = np.mean(PRmatrix_final_1)
var_final_1 = np.var(PRmatrix_final)
std_final_1 = np.sqrt(var_final_1)
itr_final_1= np.mean(PRmatrix_itr_final_1)
itr_var_final_1 = np.var(PRmatrix_itr_final_1)
itr_std_final_1 = np.sqrt(itr_var_final_1)


####CNN+FBCCA_new
PRmatrix_final_3 = PRmatrix_final_3 * 100
acc_final_3 = np.mean(PRmatrix_final_3)
var_final_3 = np.var(PRmatrix_final)
std_final_3 = np.sqrt(var_final_3)
itr_final_3= np.mean(PRmatrix_itr_final_3)
itr_var_final_3 = np.var(PRmatrix_itr_final_3)
itr_std_final_3 = np.sqrt(itr_var_final_3)

####CNN+FBTTCCA_new
PRmatrix_final_2 = PRmatrix_final_2 * 100
acc_final_2 = np.mean(PRmatrix_final_2)
var_final_2 = np.var(PRmatrix_final_2)
std_final_2 = np.sqrt(var_final_2)
itr_final_2= np.mean(PRmatrix_itr_final_2)
itr_var_final_2 = np.var(PRmatrix_itr_final_2)
itr_std_final_2 = np.sqrt(itr_var_final_2)


####CNN+FBTTCCA+CCA_new
PRmatrix_final_4 = PRmatrix_final_4 * 100
acc_final_4 = np.mean(PRmatrix_final_4)
var_final_4 = np.var(PRmatrix_final_4)
std_final_4 = np.sqrt(var_final_4)
itr_final_4= np.mean(PRmatrix_itr_final_4)
itr_var_final_4 = np.var(PRmatrix_itr_final_4)
itr_std_final_4 = np.sqrt(itr_var_final_4)

####CNN+FBTTCCA_new+CCA
PRmatrix_final_5 = PRmatrix_final_5 * 100
acc_final_5 = np.mean(PRmatrix_final_5)
var_final_5 = np.var(PRmatrix_final_5)
std_final_5 = np.sqrt(var_final_5)
itr_final_5= np.mean(PRmatrix_itr_final_5)
itr_var_final_5 = np.var(PRmatrix_itr_final_5)
itr_std_final_5 = np.sqrt(itr_var_final_5)


####CNN+FBTTCCA
PRmatrix_final_6 = PRmatrix_final_6 * 100
acc_final_6 = np.mean(PRmatrix_final_6)
var_final_6 = np.var(PRmatrix_final_6)
std_final_6 = np.sqrt(var_final_6)
itr_final_6= np.mean(PRmatrix_itr_final_6)
itr_var_final_6 = np.var(PRmatrix_itr_final_6)
itr_std_final_6 = np.sqrt(itr_var_final_6)

####total
PRmatrix_final = PRmatrix_final * 100
acc_final = np.mean(PRmatrix_final)
var_final = np.var(PRmatrix_final)
std_final = np.sqrt(var_final)
itr_final= np.mean(PRmatrix_itr_final)
itr_var_final = np.var(PRmatrix_itr_final)
itr_std_final = np.sqrt(itr_var_final)

### original
log_write_test_final.write('the acc  of FBTTCCA_new+CCA_new is:' + "\n" + str(PRmatrix) + "\n")
log_write_test_final.write('the itr  of FBTTCCA_new+CCA_new is:' + "\n" + str(PRmatrix_itr) + "\n")
log_write_test_final.write('the mean acc of FBTTCCA_new+CCA_new is:' + str(acc) + "+-" + str(std / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBTTCCA_new+CCA_new is:' + str(itr_mean) + "+-" + str(itr_std / np.sqrt(len(name))) + "\n")

log_write_test_final.write('the acc  of FBCCA_new is:' + "\n" + str(PRmatrix_CCA) + "\n")
log_write_test_final.write('the itr  of FBCCA_new is:' + "\n" + str(PRmatrix_itr_CCA) + "\n")
log_write_test_final.write('the mean acc of FBCCA_new is:' + str(acc_CCA) + "+-" + str(std_CCA / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBCCA_new is:' + str(itr_mean_CCA) + "+-" + str(itr_std_CCA / np.sqrt(len(name))) + "\n")

### gen_2
log_write_test_final.write('the acc  of FBTTCCA_new is:' + "\n" + str(PRmatrix_TTCCA) + "\n")
log_write_test_final.write('the itr  of FBTTCCA_new  is:' + "\n" + str(PRmatrix_itr_TTCCA) + "\n")
log_write_test_final.write('the mean acc of FBTTCCA_new is:' + str(acc_TTCCA) + "+-" + str(std_TTCCA / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of FBTTCCA_new is:' + str(itr_TTCCA) + "+-" + str(itr_std_TTCCA / np.sqrt(len(name))) + "\n")

### CNN
log_write_test_final.write('the acc  of CNN is:' + "\n" + str(PRmatrix_CNN) + "\n")
log_write_test_final.write('the itr  of CNN is:' + "\n" + str(PRmatrix_itr_CNN) + "\n")
log_write_test_final.write('the mean acc of CNN is:' + str(acc_CNN) + "+-" + str(std_CNN / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN is:' + str(itr_CNN) + "+-" + str(itr_std_CNN / np.sqrt(len(name))) + "\n")

### CNN+FBCCA
log_write_test_final.write('the acc  of CNN+FBCCA is:' + "\n" + str(PRmatrix_final_1) + "\n")
log_write_test_final.write('the itr  of CNN+FBCCA is:' + "\n" + str(PRmatrix_itr_final_1) + "\n")
log_write_test_final.write('the mean acc of CNN+FBCCA is:' + str(acc_final_1) + "+-" + str(std_final_1 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBCCA is:' + str(itr_final_1) + "+-" + str(itr_std_final_1 / np.sqrt(len(name))) + "\n")


### CNN+FBCCA
log_write_test_final.write('the acc  of CNN+FBCCA_new is:' + "\n" + str(PRmatrix_final_3) + "\n")
log_write_test_final.write('the itr  of CNN+FBCCA_new is:' + "\n" + str(PRmatrix_itr_final_3) + "\n")
log_write_test_final.write('the mean acc of CNN+FBCCA_new is:' + str(acc_final_3) + "+-" + str(std_final_3 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBCCA_new is:' + str(itr_final_3) + "+-" + str(itr_std_final_3 / np.sqrt(len(name))) + "\n")

### CNN+FBTTCCA_new
log_write_test_final.write('the acc  of CNN+FBTTCCA_new is:' + "\n" + str(PRmatrix_final_2) + "\n")
log_write_test_final.write('the itr  of CNN+FBTTCCA_new is:' + "\n" + str(PRmatrix_itr_final_2) + "\n")
log_write_test_final.write('the mean acc of CNN+FBTTCCA_new is:' + str(acc_final_2) + "+-" + str(std_final_2 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBTTCCA_new is:' + str(itr_final_2) + "+-" + str(itr_std_final_2 / np.sqrt(len(name))) + "\n")


### CNN+FBTTCCA_new+CCA_new
log_write_test_final.write('the acc  of CNN+FBTTCCA_new+CCA_new is:' + "\n" + str(PRmatrix_final_4) + "\n")
log_write_test_final.write('the itr  of CNN+FBTTCCA_new+CCA_new is:' + "\n" + str(PRmatrix_itr_final_4) + "\n")
log_write_test_final.write('the mean acc of CNN+FBTTCCA_new+CCA_new is:' + str(acc_final_4) + "+-" + str(std_final_4 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBTTCCA_new+CCA_new is:' + str(itr_final_4) + "+-" + str(itr_std_final_4 / np.sqrt(len(name))) + "\n")

### CNN+FBTTCCA_new+CCA
log_write_test_final.write('the acc  of CNN+FBTTCCA_new+CCA is:' + "\n" + str(PRmatrix_final_5) + "\n")
log_write_test_final.write('the itr  of CNN+FBTTCCA_new+CCA is:' + "\n" + str(PRmatrix_itr_final_5) + "\n")
log_write_test_final.write('the mean acc of CNN+FBTTCCA_new+CCA is:' + str(acc_final_5) + "+-" + str(std_final_5 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBTTCCA_new+CCA is:' + str(itr_final_5) + "+-" + str(itr_std_final_5 / np.sqrt(len(name))) + "\n")


### CNN+FBTTCCA
log_write_test_final.write('the acc  of CNN+FBTTCCA is:' + "\n" + str(PRmatrix_final_6) + "\n")
log_write_test_final.write('the itr  of CNN+FBTTCCA is:' + "\n" + str(PRmatrix_itr_final_6) + "\n")
log_write_test_final.write('the mean acc of CNN+FBTTCCA is:' + str(acc_final_6) + "+-" + str(std_final_6 / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of CNN+FBTTCCA is:' + str(itr_final_6) + "+-" + str(itr_std_final_6 / np.sqrt(len(name))) + "\n")

### final
log_write_test_final.write('the acc  of final is:' + "\n" + str(PRmatrix_final) + "\n")
log_write_test_final.write('the itr  of final is:' + "\n" + str(PRmatrix_itr_final) + "\n")
log_write_test_final.write('the mean acc of final is:' + str(acc_final) + "+-" + str(std_final / np.sqrt(len(name))) + "\n")
log_write_test_final.write('the mean itr of final is:' + str(itr_final) + "+-" + str(itr_std_final / np.sqrt(len(name))) + "\n")
# python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 main_gan.py
