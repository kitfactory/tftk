import tensorflow as tf

from abc import ABC, abstractclassmethod
import tensorflow as tf


class AbstractClassificationModel(ABC):

    def __init__(self):
        pass

    @abstractclassmethod
    def __get_base_model(cls,input:(int,int,int),include_top:bool,weighs:str,**kwargs)->tf.keras.Model:
        raise NotImplementedError()

    @classmethod
    def get_model(cls, input_shape:(int,int,int), include_top=False, weights=None, classes=1000, **kwargs)-> tf.keras.Model:
        base = cls.__get_base_model(input,include_top, weights, kwargs)


class SimpleClassificationModel(AbstractClassificationModel):
    def __init__(self):
        pass
    
    @classmethod
    def __get_base_model(cls,input_shape:(int,int,int),include_top:bool,weighs:str,**kwargs) -> tf.keras.Model:

        inputs = tf.keras.Input(shape=input_shape,name='image')
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
    def __get_base_model(cls, h:int,w:int,c:int,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(w,h,c),pooling="max")
        return model

class ResNet50V2(AbstractClassificationModel):

    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50V2(include_top=False, weights=None, input_shape=(w,h,c),pooling="max")
        return model




class SoftmaxClassifyModel():
    """単純なSoftmaxによる分類モデルクラス

    """
    def __init__(self):
        pass

    @classmethod
    def get_classify_model(cls, base_model:tf.keras.Model, classes:int)->tf.keras.Model:        
        input = base_model.input
        last = base_model.output
        x = tf.keras.layers.Flatten(name='classify-1')(last)
        x = tf.keras.layers.Dense(classes)(x)
        x = tf.keras.layers.Activation('softmax')(x)
        return tf.keras.Model(input,x)

