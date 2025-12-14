import pickle
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

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
    
    
def show_results(history):
    trs  = [history.history['accuracy'], history.history['loss']]
    vals = [history.history['val_accuracy'], history.history['val_loss']]
    titles = ["Accuracy", "Loss"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax_idx, ax in enumerate(axes.flat):
        ax.plot(trs[ax_idx], label='train', color="#1f77b4")
        ax.plot(vals[ax_idx], label='validation', color="#ff7f0e")
        ax.set_title(titles[ax_idx])
        ax.set_xlabel("Epoch")
        ax.grid(linestyle=':', alpha=0.7)
        ax.legend()
        
    fig.tight_layout()
    plt.show()