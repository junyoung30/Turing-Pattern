import pickle
import tensorflow as tf
import numpy as np
import random


def open_pickle(fpath):
    with open(fpath, 'rb') as file:
        odict = pickle.load(file)
    
    return odict

def save_pickle(data, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def set_global_seed(seed):
    tf.random.set_seed(seed) # mini-Batch Shuffling Seed
    np.random.seed(seed) # ImageDataGenerator; Data AUG Ordering Fix
    random.seed(seed)    # ImageDataGenerator; Data AUG Ordering Fix