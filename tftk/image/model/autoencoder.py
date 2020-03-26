__all__ = ['ConvAutoEncoder']

import tensorflow as tf
from abc import ABC, abstractmethod, abstractclassmethod

class AbstractAutoEncoderModel(ABC):

    @abstractclassmethod
    def get_model(cls, input_shape:(int,int,int), **kwargs):
        raise NotImplementedError()

class ConvAutoEncoder(AbstractAutoEncoderModel):
    def __init__(self):
        pass

    @classmethod
    def get_model(cls, input_shape, layers=10):
        input_img = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='encoded')(x)

        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = tf.keras.Model(input_img, decoded)
        return autoencoder
