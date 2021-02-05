from abc import ABC, abstractclassmethod, abstractmethod
import tensorflow as tf

class AbstractRepresentationModel(ABC):

    def __init__(self):
        pass

    @abstractclassmethod
    def get_base_model(cls,input:(int,int,int),weights:str,**kwargs)->tf.keras.Model:
        raise NotImplementedError()

    @classmethod
    def get_representation_model(cls, input_shape:(int,int,int),weights=None, classes=1000, **kwargs)-> tf.keras.Model:
        base = cls.get_base_model(input_shape, weights, **kwargs)

class SimpleRepresentationModel(ABC):

    def __init__(self):
        pass

    @classmethod
    def get_base_model(cls,input:(int,int,int),weights:str,**kwargs)->tf.keras.Model:
        input_layer = tf.keras.layers.Input(shape=input)
        x = tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3),padding='same')(input_layer)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3),padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3),padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        return tf.keras.Model(inputs=[input_layer],outputs=[x])

    @classmethod
    def get_representation_model(cls, input_shape:(int,int,int),weights=None, **kwargs)-> tf.keras.Model:
        base = cls.get_base_model(input_shape, weights, **kwargs)
        return base

def norm(x):
    x = tf.keras.backend.l2_normalize(x)
    return x

def add_projection_layers(model:tf.keras.Model, normalize=True)->tf.keras.Model:
    print(model.layers)
    length = len(model.layers)
    print( "model length ", length )
    output_layer:tf.keras.layers.Layer = model.layers[length-1]
    input_layer:tf.keras.layers.Layer = model.layers[0]

    print(output_layer.output)
    flatten = tf.keras.layers.Flatten()(output_layer.output)
    dense1 = tf.keras.layers.Dense(128)(flatten)
    dense1 = tf.keras.layers.Activation('relu')(flatten)
    dense2 = tf.keras.layers.Dense(16)(dense1)
    if normalize == True:
        dense2 = tf.keras.layers.Lambda(norm)(dense2)
    model = tf.keras.Model(inputs=input_layer.input,outputs=dense2)
    return model







