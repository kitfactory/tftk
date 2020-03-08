import tensorflow as tf

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

