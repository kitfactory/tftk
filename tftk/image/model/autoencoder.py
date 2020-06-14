import tensorflow as tf
from abc import ABC, abstractmethod, abstractclassmethod

class AbstractAutoEncoderModel(ABC):

    @abstractclassmethod
    def get_model(cls, input_shape:(int,int,int), **kwargs):
        raise NotImplementedError()

class SimpleAutoEncoderModel(AbstractAutoEncoderModel):
    def __init__(self):
        pass

    @classmethod
    def get_model(cls, input_shape, **kwargs):
        h, w, c = input_shape
        input_img = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
        # x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        # encoded = tf.keras.layers.AveragePooling2D((2, 2), padding='same', name='encoded')(x)

        # x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
        # # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        # x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same',dtype='float32')(x)
        decoded = tf.keras.layers.Conv2D(c, (3, 3), activation='sigmoid', padding='same',dtype='float32')(x)

        autoencoder = tf.keras.Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder


class ResNetAutoEncoderModel(AbstractAutoEncoderModel):
    def __init__(self):
        pass

    @classmethod
    def get_model(cls, input_shape, **kwargs):
        input_img = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.AveragePooling2D((2, 2), padding='same', name='encoded')(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
        # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        # x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2,2) , padding='same')(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = tf.keras.Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder




class SSIMAutoEncoderModel(AbstractAutoEncoderModel):
    def __init__(self):
        pass

    @classmethod
    def get_model(cls, input_shape, **kwargs):
        h,w,c = input_shape
        input_img = tf.keras.layers.Input(shape=input_shape)
        
        D =8
        if c == 3:
            depth = 2
        else:
            depth = 1

        # encoder       
        x = tf.keras.layers.Conv2D(32 * depth, (4, 4), strides=2, padding='same')(input_img) # 64,64,32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(32 * depth, (4, 4), strides=2, padding='same')(x) # 32,32,32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(32 * depth, (3, 3), strides=1, padding='same')(x) # 32,32,32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(64 * depth, (4, 4), strides=2, padding='same')(x) # 16,16*64*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(64 * depth, (3, 3), strides=1, padding='same')(x) # 16,16*64*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(128 * depth, (4, 4), strides=2, padding='same')(x) # 8,8 128*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(64 * depth, (3, 3), strides=1, padding='same')(x)  # 8,8 64*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(32 * depth, (3, 3), strides=1, padding='same')(x)  # 8,8 32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(D*depth, (8, 8), strides=1, padding='same')(x)  # 8 * 8 * D
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # decoder
        x = tf.keras.layers.Conv2D(32*depth, (3,3), strides=1, padding='same')(x) # 8, 8 , 32
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(64*depth, (3,3), strides=1, padding='same')(x) # 8, 8, 64
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(128*depth, (4,4), strides=2, padding='same')(x) # 8,8, 128
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(64 * depth, (3, 3), strides=1, padding='same')(x) # 16,16,64
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(64*depth, (4,4), strides=2, padding='same')(x) # 32, 32, 64
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(32 * depth, (3, 3), strides=1, padding='same')(x) # 32,32,32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(32*depth, (4,4), strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2DTranspose(32*depth, (4,4), strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), strides=1, padding='same')(x) # 32,32,32*d
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        autoencoder = tf.keras.Model(input_img, x)
        autoencoder.summary()
        return autoencoder
