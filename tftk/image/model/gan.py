from abc import ABC, abstractclassmethod
import tensorflow as tf

class AbstractGANModel(ABC):

    @abstractclassmethod
    def get_generator_model(cls, latent_shape=(32),output_shape=(32,32,3), **kwargs):
        pass

    @abstractclassmethod
    def get_discriminator_model(cls, input_shape=(32,32,3),latent_shape=(32), **kwargs):
        pass


class DCGAN(AbstractGanModel):

    @classmethod
    def get_generator_model(cls, latent_shape=(32),output_shape=(32,32,3), **kwargs):
        h, w, c = output_shape

        input = tf.keras.Input(shape=(latent_shape,))
        x = tf.keras.layers.Dense(128*h*w)(x)
        x = tf.keras.layers.LeakyReLU()
        x = tf.keras.layers.Reshape((h,w,128))(x) # 16,16,128

        x = tf.keras.layers.Conv2D(256,5, padding='same') # 16,16,256
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(256,4, stride=2, padding='same')  # 32,32,256
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(256,5, padding='same') # 32,32,256
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(256,5, padding='same')  # 32,32,256
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Con2D(c,7,activation='tanh',padding='same') # 32,32,3

        generator = tf.keras.Model(input,x)
        generator.summary()

        return generator

    @classmethod
    def get_discriminator_model(cls, input_shape=(32,32,3),latent_shape=(32), **kwargs):
        input = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(128,3)(input)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(128,4,strides=2)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(128,4,strides=2)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(128,4,strides=2)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

        discriminator = tf.keras.Model(input,x)
        discriminator.summary()

        return discriminator
