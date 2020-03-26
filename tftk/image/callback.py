from abc import ABC, abstractclassmethod
import tensorflow as tf

class Callback(ABC):
    def __init__(self):
        pass

class HandyCallback():
    @classmethod
    def get_callbacks(cls, tensorboard_log=None, save_weights="./model.hdf5", **kwargs):
        callbacks = []
        if tensorboard_log is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log))
        if save_weights is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_weights,monitor="val_acc",save_best_only=True,save_weights_only=True))
        return callbacks

