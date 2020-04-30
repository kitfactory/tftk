import tensorflow as tf

from tftk.image.model import AbstractClassificationModel

from tftk.image.model.small_resnet_keras_contrib import ResNet18

def shortcut(input:tf.keras.layers.Layer,output:tf.keras.layers.Layer)->tf.keras.layers.Layer:
    if input.shape != output.shape: # ショートカットにダウンサンプルが必要
        x = tf.keras.layers.AveragePooling2D(input) # ResNet-D
        x = tf.keras.layers.Conv2D() # down sample shortcut path
    return x

def get_layer_name(layer:int, block:int, elem:str):
    return "{}_{}_{}".format(elem,layer,block)

def get_residual_unit_v2(x:tf.keras.layers.Layer, layer:int, block:int, filters:int, kernel_size:(int,int)):
    # ResNet V2は、http://arxiv.org/pdf/1603.05027v2.pdf
    dilation_rate = (1,1)
    ki = "he_normal"
    kr = tf.keras.regularizers.l2(1e-4)

    if block!=1:
        # ダウンサンプルがいらない
        strides = (1,1)
        x = tf.keras.layers.BatchNormalization(name=get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
        x = tf.keras.layers.Activation("relu", name=get_layer_name(layer=layer,block=block,elem="relu"))(x)
        x = tf.keras.layers.("relu", name=get_layer_name(layer=layer,block=block,elem="relu"))(x)
    elif layer == 2 and block==1:
        # ダウンサンプル、BN/Reluが要らない
        strides = (1,1)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer=ki,
                      kernel_regularizer=kr,
                      name=get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
    else:
        # BN/Reluが要る、ダウンサンプルする。
        strides = (2,2)
        x = tf.keras.layers.BatchNormalization(name=get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
        x = tf.keras.layers.Activation("relu", name=get_layer_name(layer=layer,block=block,elem="relu"))(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer=ki,
                      kernel_regularizer=kr,
                      name=get_layer_name(layer=layer,block=block,elem="conv2d"))(x)

def basic_block(input:tf.keras.layers.Layer,  layer:int, filters:int, repetitions:int , downsample:bool):

    if layer == 2:
        # ダウンサンプルはmax pooling
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same", name=get_layer_name(layer,1,"maxpooling"))(x)

    x = get_residual_unit_v2(x=x, layer=2,block=1, filters=filters)

    x = get_residual_unit_v2(x=x, layer=2,block=2, filters=filters)
    
    # x = tf.keras.
    s = shortcut(input,x)
    x = tf.keras.layers.Add([x,s]) #  shortcut path
    return x

def get_resnet(input_shape)->tf.keras.Model:

    repeatations = [2,2,2,2]
    #     18 : 
    #     34: [3,4,6,3],
    #     50: [3,6,4,3],
    #     101: [3,6,4,3],
    #     152: [3,6,4,3],
    # }

    bottleneck_func = basic_block
    #     18: basic_block,
    #     34: basic_block,
    #     50: bottleneck, 
    #     101: bottleneck,
    #     152: bottleneck,
    # }

    activation = 'relu'
    initial_kernel_size=(7, 7)
    initial_strides=(2, 2)
    initial_pooling='max'
    initial_filters=64
    activation='relu'

    kr = tf.keras.regularizers.l2(1e-11)
    ki = 'he_normal'

    # Conv1
    img_input = tf.keras.Input(shape=input_shape, name='input')
    x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=initial_strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same")(img_input)
    x = tf.keras.layers.BatchNormalization(name='conv1_batchnorm')(x)
    x = tf.keras.layers.Activation("relu", name='conv1_relu')(x)

    filters = initial_filters
    for r in range(repeatations):
         for idx in range(r):
             x = basic_block(x, layer=(r + 2), filter=filters)
    
    model = tf.keras.Model(img_input, x)
    model.summary()
    return model

    # Conv2-4
    exit()



    #Conv2-4


if __name__ == '__main__':

    model = ResNet18.get_base_model(input_shape=(224,224,3),include_top=False)
    model.summary()

    model = get_resnet(input_shape=(224,224,3))
