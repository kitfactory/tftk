from abc import ABC, abstractclassmethod
import tensorflow as tf


class AbstractClassificationModel(ABC):

    def __init__(self):
        pass

    @abstractclassmethod
    def get_base_model(cls, h:int, w:int, c:int, **kwargs)-> tf.keras.Model:
        pass


class SimpleBaseModel(AbstractClassificationModel):
    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int,**kwargs) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(h,w,c,),name='image')
        x = tf.keras.layers.Conv2D(32,(3,3),padding='same')(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(32,(3,3),padding='same')(x) 
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        x = tf.keras.layers.Conv2D(32,(3,3),padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(32,(3,3),padding='same')(x) 
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

        model = tf.keras.Model(inputs=inputs,outputs=x)
        return model


class ResNet50(AbstractClassificationModel):
    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(w,h,c),pooling="max")
        return model

class ResNet50V2(AbstractClassificationModel):

    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50V2(include_top=False, weights=None, input_shape=(w,h,c),pooling="max")
        return model


