import os
import time
import pickle
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from .model import CNN
from .dataset import get_dataset, scaling
from .utils import set_global_seed, save_pickle



def prepare_data(
    DirData, 
    DataSeed
):
    X, labels = get_dataset(DirData)

    X = np.expand_dims(X, axis=-1)
    X = scaling(X)

    y = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=DataSeed
    )
    
    return X_train, X_test, y_train, y_test

def train_model(
    X_train, 
    X_test, 
    y_train, 
    y_test,
    NumBlocks,
    NumConv,
    NumDense,
    WeightSeed,
    AugFLAG,
    LR,
    BS,
    EPOCHS
):
    set_global_seed(1234)
    
    optimizer = SGD(learning_rate=LR)
    
    datagen = ImageDataGenerator(horizontal_flip=AugFLAG, vertical_flip=AugFLAG)
    datagen.fit(X_train)
    
    input_shape = X_train.shape[1:]
    output_dim = y_train.shape[1]
    
    model = CNN(
        input_shape=input_shape,
        num_classes=output_dim,
        num_blocks=NumBlocks,
        num_conv_filters=NumConv,
        dense_units=NumDense,
        seed=WeightSeed
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BS),
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        shuffle=True,
        verbose=False,
    )
    
    return model, history


def save_results(
    model, 
    history, 
    save_dir, 
    model_name
):
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, model_name + ".keras")
    model.save(model_save_path)

    meta = {"history": history.history}
    meta_save_path = os.path.join(save_dir, model_name + ".pkl")
    save_pickle(meta, meta_save_path)

    print(f"Save Model: {model_save_path}")


