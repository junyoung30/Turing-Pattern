import os
import numpy as np
import pickle

from .utils import open_pickle


def get_dataset(DirData):
    
    label2idx = {
        "Sdot": 0, "Ldot": 1, "Line": 2, "Mdot": 3,
    }
    
    FileNames = [f for f in os.listdir(DirData) if f.endswith('.pkl')]
    FileNames = sorted(FileNames, key=lambda x: int(x.split('.')[0]))

    X, labels = [], []
    for FileName in FileNames:
        fpath = os.path.join(DirData, FileName)        
        data = open_pickle(fpath)
    
        vv = data['vlist'][-1]
        bb = data['pattern']['label']
        bb = label2idx[bb]
        
        patches = vv.reshape(4,64,4,64).transpose(0,2,1,3).reshape(16,64,64)
        
        X.append(patches)
        labels.append(bb*np.ones(16))
    
    X = np.array(X).reshape(-1,64,64)
    labels = np.array(labels).reshape(-1)
    return X, labels


def scaling(X):
    min_value = np.min(X, axis=(1, 2), keepdims=True)
    max_value = np.max(X, axis=(1, 2), keepdims=True)
    
    X_scaled = (X-min_value)/(max_value-min_value)
    return X_scaled