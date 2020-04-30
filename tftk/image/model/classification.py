import tensorflow as tf

from abc import ABC, abstractclassmethod, abstractmethod
import tensorflow as tf

class AbstractClassificationModel(ABC):

    def __init__(self):
        pass

    @abstractclassmethod
    def get_base_model(cls,input:(int,int,int),include_top:bool,weighs:str)->tf.keras.Model:
        raise NotImplementedError()

    @classmethod
    def get_model(cls, input_shape:(int,int,int), include_top=False, weights=None, classes=1000, **kwargs)-> tf.keras.Model:
        base = cls.get_base_model(input_shape,include_top, weights)
        model = ClassificationModel.get_classify_model(base,classes)
        return model

class SimpleClassificationModel(AbstractClassificationModel):
    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls,input_shape:(int,int,int),include_top:bool,weighs:str,**kwargs) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape,name='image')
        x = tf.keras.layers.Conv2D(32,(3,3))(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)

        x = tf.keras.layers.Conv2D(64,(3,3))(x) 
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)

        x = tf.keras.layers.Conv2D(128,(3,3))(x) 
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)

        x = tf.keras.layers.Conv2D(128,(3,3))(x) 
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        model = tf.keras.Model(inputs=inputs,outputs=x)
        return model


class ResNet50(AbstractClassificationModel):
    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls,input_shape:(int,int,int),include_top:bool,weighs:str) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape,pooling="avg")
        return model

class ResNet50V2(AbstractClassificationModel):

    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls,input_shape:(int,int,int),include_top:bool,weighs:str,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.resnet50.ResNet50V2(include_top=False, weights=None, input_shape=input_shape,pooling="avg")
        return model

class MobileNetV2(AbstractClassificationModel):

    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls,input_shape:(int,int,int),include_top:bool,weighs:str,**kwargs) -> tf.keras.Model:
        model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape,pooling="avg")
        return model


class ClassificationModel():
    """単純なSoftmaxによる分類モデルクラス

    """
    def __init__(self):
        pass

    @classmethod
    def get_classify_model(cls, base_model:tf.keras.Model, classes:int)->tf.keras.Model:        
        input = base_model.input
        last = base_model.output

        x = tf.keras.layers.Flatten(name='classify-1')(last)
        x = tf.keras.layers.Dense(512, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation('relu')(x)

        if classes != 2:
            x = tf.keras.layers.Dense(classes,dtype='float32',kernel_initializer='he_normal')(x)
            x = tf.keras.layers.Activation('softmax',dtype='float32')(x)
        else:
            x = tf.keras.layers.Dense(classes,dtype='float32',kernel_initializer='he_normal')(x)
            x = tf.keras.layers.Activation('sigmoid',dtype='float32')(x)
        return tf.keras.Model(input,x)

