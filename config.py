import os
from datetime import datetime
import numpy as np
from scipy.io import loadmat

class Config(object):
    def __init__(self):

        self.Dataset=2#0 for UCSD, 1 for benchmark, 2 for BETA
        self.LOOC = 0 #LOOC shot: 1-14 for UCSD, 1-5 for benchmark, 1-3 for BETA
        self.sample_length=0.5
        self.Nh=4
        self.Nm=3
        self.n_epochs=100
        self.batch_size=128

        if self.Dataset==0:
            self.C = 8 
            self.rfs=256
            self.dropout=0.95
            self.num_class = 12
            self.nBlock=15
            self.list_freqs = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75,14.75]).T  # list of stimulus frequencies
            self.list_phase = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi  # list of stimulus phase    

        if self.Dataset==1:
            self.C = 9 
            self.rfs=250
            self.dropout=0.95 
            self.num_class = 40
            self.nBlock=6
            self.list_freqs = loadmat("/home/wc/Code/SSVEP_data/Bench/Freq_Phase.mat")['freqs'][0]
            self.list_phase=np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5])* np.pi 

        if self.Dataset==2:
            self.C = 9 
            self.rfs=250
            self.dropout=0.95  
            self.num_class = 40
            self.nBlock=4
            self.list_freqs = loadmat("/home/wc/Code/SSVEP_data/BETA/Freqs_Beta.mat")['freqs'][0]
            self.list_phase = np.array([1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1]) * np.pi

        self.T = int(self.sample_length*self.rfs)
        self.save_path = os.path.abspath(os.curdir)+'/result/'
            
config = Config()
