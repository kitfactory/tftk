import tensorflow as tf
from typing import Tuple

from tftk.image.model import AbstractClassificationModel
# from tftk.image.model import ResNetD18

# class ResNetBuilder():

#     ROW_AXIS = 1
#     COL_AXIS = 2
#     CHANNEL_AXIS = 3

#     def __init__(self):
#         pass

#     def get_layer_name(self, layer:int, block:int, elem:str):
#         name = "{}_{}_{}".format(elem,layer,block)
#         print(name)
#         return name

#     def shortcut(self, input:tf.keras.layers.Layer,output:tf.keras.layers.Layer,layer:int, block:int)->tf.keras.layers.Layer:
#         # ResNet-D
#         x = tf.keras.layers.AveragePooling2D(strides=(1,1),padding="same",name=self.get_layer_name(layer,block,"avg"))(input)

#         # Expand channels of shortcut to match residual.
#         # Stride appropriately to match residual (width, height)
#         # Should be int if network architecture is correctly configured.
#         input_shape = tf.keras.backend.int_shape(input)
#         output_shape = tf.keras.backend.int_shape(output)
#         stride_width = int(round(input_shape[ResNetBuilder.ROW_AXIS] / output_shape[ResNetBuilder.ROW_AXIS]))
#         stride_height = int(round(input_shape[ResNetBuilder.COL_AXIS] / output_shape[ResNetBuilder.COL_AXIS]))
#         equal_channels = input_shape[ResNetBuilder.CHANNEL_AXIS] == output_shape[ResNetBuilder.CHANNEL_AXIS]
#         shortcut = x
#         # 1 X 1 conv if shape is different. Else identity.
#         if stride_width > 1 or stride_height > 1 or not equal_channels:
#             shortcut = tf.keras.layers.Conv2D(filters=output_shape[ResNetBuilder.CHANNEL_AXIS],
#                 kernel_size=(1, 1),
#                 strides=(stride_width, stride_height),
#                 padding="valid",
#                 kernel_initializer="he_normal",
#                 kernel_regularizer=tf.keras.regularizers.l2(0.0001),
#                 name=self.get_layer_name(layer,block,"shortcut_conv"))(x)
        
#         shortcut = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,
#                 name=self.get_layer_name(layer,block,"shortcut_batchnorm"))(shortcut)

#         x = tf.keras.layers.Add()([output,shortcut]) #  shortcut path
#         return x


#     def get_residual_unit_v2(self, x:tf.keras.layers.Layer, layer:int, block:int, filters:int, kernel_size:(int,int)):
#         # ResNet V2は、http://arxiv.org/pdf/1603.05027v2.pdf
#         dilation_rate = (1,1)
#         ki = "he_normal"
#         kr = tf.keras.regularizers.l2(1e-4)
#         print("get residual unit layer",layer,"block",block, "Tensor",x)

#         if block!=1:
#             # ダウンサンプルがいらない
#             strides = (1,1)
#             x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,name=self.get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
#             x = tf.keras.layers.Activation("relu", name=self.get_layer_name(layer=layer,block=block,elem="relu"))(x)
#             x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
#                       strides=strides, padding="same",
#                       dilation_rate=dilation_rate,
#                       kernel_initializer=ki,
#                       kernel_regularizer=kr,
#                       name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
#         elif layer == 2 and block==1:
#             # MaxPoolingでダウンサンプル、BN/Reluが要らない
#             x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same")(x)
#             strides = (1,1)
#             x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
#                       strides=strides, padding="same",
#                       dilation_rate=dilation_rate,
#                       kernel_initializer=ki,
#                       kernel_regularizer=kr,
#                       name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
#         else:
#             # BN/Reluが要る、ダウンサンプルする。
#             strides = (2,2)
#             x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,name=self.get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
#             x = tf.keras.layers.Activation("relu", name=self.get_layer_name(layer=layer,block=block,elem="relu"))(x)
#             x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
#                       strides=strides, padding="same",
#                       dilation_rate=dilation_rate,
#                       kernel_initializer=ki,
#                       kernel_regularizer=kr,
#                       name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
#         return x

#     def basic_block(self, input:tf.keras.layers.Layer, layer:int, filters:int):
#         print("basic block layer",layer)
#         x = self.get_residual_unit_v2(x=input,layer=layer,block=1, filters=filters, kernel_size=(3,3))
#         x = self.shortcut(input,x,layer,1)

#         x = self.get_residual_unit_v2(x=x,layer=layer,block=2, filters=filters, kernel_size=(3,3))
#         x = self.shortcut(input,x,layer,2)
#         return x

#     def bottleneck_block(self, input:tf.keras.layers.Layer, layer:int, filters:int):
#         x = self.get_residual_unit_v2(x=input,layer=layer,block=1, filters=filters, kernel_size=(1,1))
#         x = self.shortcut(input,x,layer,1)

#         x = self.get_residual_unit_v2(x=input,layer=layer,block=1, filters=filters, kernel_size=(3,3))
#         x = self.shortcut(input,x,layer,1)

#         x = self.get_residual_unit_v2(x=x,layer=layer,block=2, filters=filters*4, kernel_size=(1,1))
#         x = self.shortcut(input,x,layer,2)
#         return x


#     def build_resnet_base(self, input_shape:Tuple[int,int,int], size:int)->tf.keras.Model:

#         if size not in [18,34,50,101,152]:
#             raise ValueError('Wrong ResNet Size')

#         repeatations_table ={
#             18: [2,2,2,2],
#             34: [3,4,6,3],
#             50: [3,6,4,3],
#             101: [3,6,23,3],
#             152: [3,8,36,3],
#         }
#         repeatations = repeatations_table[size]

#         activation = 'relu'
#         initial_kernel_size=(7, 7)
#         initial_strides=(2, 2)
#         initial_pooling='max'
#         initial_filters=64
#         activation='relu'

#         kr = tf.keras.regularizers.l2(1e-11)
#         ki = 'he_normal'

#         # Conv1
#         img_input = tf.keras.Input(shape=input_shape, name='input')
#         x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=initial_strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same",name=self.get_layer_name(1,0,"conv2d"))(img_input)
#         x = tf.keras.layers.BatchNormalization(name=self.get_layer_name(1,0,"barchnorm"))(x)
#         x = tf.keras.layers.Activation("relu", name=self.get_layer_name(1,0,"relu"))(x)

#         # Conv2-5
#         filters = initial_filters
#         for idx,r in enumerate(repeatations):
#             print("for loop  r", r, "idx", idx)
#             if size <= 34:
#                 x = self.basic_block(x, layer=(idx + 2), filters=filters)
#             else:
#                 x = self.bottleneck_block(x, layer=(idx + 2), filters=filters)
#             filters = filters*2
    
#         # Last activation
#         x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS, name=self.get_layer_name(6,0,"final-batchnorm"))(x)
#         x = tf.keras.layers.Activation("relu", name=self.get_layer_name(6,0,"final-relu"))(x)
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)

#         model = tf.keras.Model(img_input, x)
#         model.summary()
#         return model

# class ResNet18(AbstractClassificationModel):

#     def __init__(self):
#         pass

#     @classmethod
#     def get_base_model(cls,input_shape:(int,int,int),include_top:bool)->tf.keras.Model:
#         if include_top == True:
#             raise NotImplementedError("This model not support include top")
        
#         builder = ResNetBuilder()
#         return builder.build_resnet_base(input_shape,18)


# class ResNet34(AbstractClassificationModel):

#     def __init__(self):
#         pass

#     @classmethod
#     def get_base_model(cls,input_shape:(int,int,int),include_top:bool)->tf.keras.Model:
#         if include_top == True:
#             raise NotImplementedError("This model not support include top")
        
#         builder = ResNetBuilder()
#         return builder.build_resnet_base(input_shape,34)



if __name__ == '__main__':

    model = tf.keras.applications.ResNet50V2(input_shape=(224,224,3),include_top=True,classes=10,weights=None)
    model.summary()