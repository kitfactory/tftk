"""
This code is TensorFlow porting of ResNeSt.

https://github.com/zhanghang1989/ResNeSt

## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020


 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


"""

import tensorflow as tf
from typing import Tuple
from typing import List

from tftk.image.model import AbstractClassificationModel
from tftk.activation import Mish

class ResNetBuilder():

    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3

    def __init__(self):
        pass

    def get_layer_name(self, layer:int, block:int, elem:str):
        name = "{}_{}_{}".format(elem,layer,block)
        return name

    def shortcut(self, input:tf.Tensor,output:tf.Tensor,layer:int, block:int)->tf.keras.layers.Layer:
        # ResNet-D
        # http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf

        shortcut = input
        if self.resnet_d:
            shortcut = tf.keras.layers.AveragePooling2D(strides=(1,1),padding="same",name=self.get_layer_name(layer,block,"avg"))(input)
        
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = tf.keras.backend.int_shape(input)
        output_shape = tf.keras.backend.int_shape(output)
        stride_width = int(round(input_shape[ResNetBuilder.ROW_AXIS] / output_shape[ResNetBuilder.ROW_AXIS]))
        stride_height = int(round(input_shape[ResNetBuilder.COL_AXIS] / output_shape[ResNetBuilder.COL_AXIS]))
        equal_channels = input_shape[ResNetBuilder.CHANNEL_AXIS] == output_shape[ResNetBuilder.CHANNEL_AXIS]

        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = tf.keras.layers.Conv2D(filters=output_shape[ResNetBuilder.CHANNEL_AXIS],
                kernel_size=(1, 1),
                strides=(stride_width, stride_height),
                padding="valid",
                kernel_initializer="he_normal",
                # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                name=self.get_layer_name(layer,block,"shortcut_conv"))(shortcut)
        
            shortcut = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,
                 name=self.get_layer_name(layer,block,"shortcut_batchnorm"))(shortcut)
        

        x = tf.keras.layers.Add()([output,shortcut]) #  shortcut path
        return x


    def get_residual_unit_v2(self, x:tf.keras.layers.Layer, layer:int, block:int, filters:int, kernel_size:(int,int)):
        # ResNet V2、http://arxiv.org/pdf/1603.05027v2.pdf
        
        dilation_rate = (1,1)
        ki = "he_normal"
        kr = tf.keras.regularizers.l2(1e-4)

        if block!=1:
            # ダウンサンプルがいらない
            strides = (1,1)
            x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,name=self.get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(layer=layer,block=block,elem=self.activation))(x)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer=ki,
                      # kernel_regularizer=kr,
                      name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
        elif layer == 2 and block==1:
            # MaxPoolingでダウンサンプル、BN/Reluが要らない
            strides = (1,1)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer=ki,
                      # kernel_regularizer=kr,
                      name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
        else:
            # BN/Reluが要る、ダウンサンプルする。
            strides = (2,2)
            x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS,name=self.get_layer_name(layer=layer,block=block,elem="batchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(layer=layer,block=block,elem=self.activation))(x)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding="same",
                      dilation_rate=dilation_rate,
                      kernel_initializer=ki,
                      # kernel_regularizer=kr,
                      name=self.get_layer_name(layer=layer,block=block,elem="conv2d"))(x)
        return x

    def basic_block(self, input:tf.Tensor, layer:int, filters:int):
        x = self.get_residual_unit_v2(x=input,layer=layer,block=1, filters=filters, kernel_size=(3,3))
        x2 = self.shortcut(input,x,layer,1)

        x = self.get_residual_unit_v2(x=x2,layer=layer,block=2, filters=filters, kernel_size=(3,3))
        x = self.shortcut(x2,x,layer,2)
        return x

    def bottleneck_block(self, input:tf.Tensor, layer:int, filters:int):
        x = self.get_residual_unit_v2(x=input,layer=layer,block=1, filters=filters, kernel_size=(1,1))
        x2 = self.shortcut(input,x,layer,1)

        x = self.get_residual_unit_v2(x=x2,layer=layer,block=2, filters=filters, kernel_size=(3,3))
        x3 = self.shortcut(x2,x,layer,2)

        x = self.get_residual_unit_v2(x=x3,layer=layer,block=3, filters=filters*4, kernel_size=(1,1))
        x = self.shortcut(x3,x,layer,3)
        return x

    def split_attention_simplified_block(self, input:tf.Tensor, layer:int, filters:int):
        x = self.get_split_attention_unit(input=input,layer=layer,block=1, filters=filters, downsample=True)
        x2 = self.shortcut(input,x,layer,1)

        x = self.get_split_attention_unit(input=x2,layer=layer,block=2, filters=filters, downsample=False)
        x = self.shortcut(x2,x,layer,2)
        return x

    def split_attention_block(self, x:tf.Tensor, layer:int, filters:int):
        x = self.get_split_attention_unit(input=input,layer=layer,block=1, filters=filters, downsample=True)
        x2 = self.shortcut(input,x,layer,1)

        x = self.get_split_attention_unit(input=x2,layer=layer,block=2, filters=filters, downsample=False)
        x3 = self.shortcut(input,x,layer,2)

        x = self.get_split_attention_unit(input=x3,layer=layer,block=3, filters=filters*4, downsample=False)
        x = self.shortcut(input,x,layer,3)
        return x

    def get_split_attention_unit(self, input:tf.Tensor, layer:int, block:int, filters:int , downsample:bool=True):
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        ＊out = self.dropblock1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        ＊out = self.dropblock2(out)
        out = self.avd_layer(out):Avg Pool2D # downsample
        out = self.conv3(out)　
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        ＊　out = self.dropblock3(out) 
        out = out + residual
        out = self.relu3(out)
        """

        ki = "he_normal"
        kr = tf.keras.regularizers.l2(1e-4)

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), kernel_initializer=ki,
            # kernel_regularizer=kr, 
            padding="same", name=self.get_layer_name(layer, block ,"cnn1"))(input)
        x = tf.keras.layers.BatchNormalization(axis=-1,name=self.get_layer_name(layer, block ,"split_batchnorm1"))(x)
        # x = self.dropblock(x, prob)
        x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(layer,block,"split_{}".format(self.activation)))(x)

        if self.size > 34 or downsample == True:
            x = self.split_atteintion_conv(x,layer,block,filters)
        else:
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3),kernel_initializer=ki,
            # kernel_regularizer=kr,
            padding="same")(x)
            x = tf.keras.layers.BatchNormalization(axis=-1,name=self.get_layer_name(layer, block ,"split_batchnorm1-2"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(layer,block,"split_{}-2".format(self.activation)))(x)

        if downsample == True:
            x = tf.keras.layers.AveragePooling2D(pool_size=3,strides=(2,2),padding="same",name=self.get_layer_name(layer, block ,"avg.pooling"))(x)
            self.update_img_size(2)

        x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),kernel_initializer=ki,kernel_regularizer=kr,padding="same",name=self.get_layer_name(layer, block ,"cnn2"))(x)
        x = tf.keras.layers.BatchNormalization(axis=-1,name=self.get_layer_name(layer, block ,"split_batchnorm2"))(x)

        return x



    def split_atteintion_conv(self,input:tf.Tensor, layer:int, block:int, filters:int):
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        ＊out = self.dropblock1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        ＊out = self.dropblock2(out)
        
        out = self.avd_layer(out):Avg Pool2D
        out = self.conv3(out)　
        out = self.bn3(out)　　

        if self.downsample is not None:
            residual = self.downsample(x)

        ＊　out = self.dropblock3(out) 

        out = out + residual
        out = self.relu3(out)

        return out
        """
        residual = input
        cardinality = 1
        radix = 2
        dropout_rate = 0.1
        inter_channels = max(filters*radix//2//2, 32) # r = 2

        cardinal_branch = []
        first_cnn_channels = filters // cardinality // radix
        second_cnn_channels = filters // cardinality
        fc1_cnn_channels = inter_channels
        fc2_cnn_channels = filters * radix
        third_cnn_channels = filters

        # まとめるには？
        # 2-1は1層目は全部まとめて filters*cardinality
        # 2-2は2層目は?

        ki = "he_normal"
        kr = tf.keras.regularizers.l2(1e-4)

        radix_branch = []
        for r in range(radix):
            cardinal_branch = []
            for c in range(cardinality):
                # conv1
                x = tf.keras.layers.Conv2D(filters=first_cnn_channels,kernel_size=(1,1), padding="same",kernel_initializer=ki,kernel_regularizer=kr)(input)
                x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS)(x)
                # x = tf.keras.layers.DropBlock(dropblock rate))
                x = tf.keras.layers.Activation(self.activation)(x)

                # conv2
                x = tf.keras.layers.Conv2D(filters=second_cnn_channels,kernel_size=(3,3), padding="same",kernel_initializer=ki,kernel_regularizer=kr)(x)
                x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS)(x)
                x = tf.keras.layers.Activation(self.activation)(x)
                cardinal_branch.append(x)
            
            # concatenate
            if cardinality == 1:
                radix_group = x
            else:
                radix_group = tf.keras.layers.Concatenate()(cardinal_branch)
            radix_branch.append(radix_group)

        # Add
        gx = tf.keras.layers.Add()(radix_branch)
        
        # Global pooling
        gp = tf.keras.layers.GlobalAveragePooling2D()(gx) # チャンネルごとにPooling?
        gp = tf.keras.layers.Reshape((1,1,second_cnn_channels))(gp)

        # Keras Conv2DにgroupがないのでCardinality分にK分割し、Denseを2度実施する。
        slices = []
        slices_size = second_cnn_channels // cardinality
        for c in range(cardinality):
            layer_start = c * third_cnn_channels
            layer_end = ((c + 1) * third_cnn_channels) -1
            sx = tf.keras.layers.Lambda((lambda x:x[:, layer_start:layer_end]), output_shape=(slices_size,1,1))(gp)
            sx = tf.keras.layers.Conv2D(filters=fc1_cnn_channels,kernel_size=(1,1), padding="same",kernel_initializer=ki,kernel_regularizer=kr)(sx)
            sx = tf.keras.layers.BatchNormalization(axis=1)(sx)
            sx = tf.keras.layers.Activation(self.activation)(sx)
            sx = tf.keras.layers.Conv2D(filters=fc2_cnn_channels,kernel_size=(1,1), padding="same",kernel_initializer=ki,kernel_regularizer=kr)(sx)
            slices.append(sx)

        if cardinality == 1:
            concat_sx = sx
        else:
            concat_sx = tf.keras.layers.Concatenate()(slices)
        # reshape to rsoftmax acceptable
        reshape_sx = tf.keras.layers.Reshape((radix,filters),name=self.get_layer_name(layer,block,"reshape"))(concat_sx)

        # get attention
        attention = self.rsoftmax(reshape_sx,radix,fc2_cnn_channels)
        def elemwise_multiply(tensors):
            x = tensors[0]
            y = tensors[1]
            c = tensors[2]
            y = tf.keras.backend.reshape(y,shape=(-1,1,1,filters))
            s =[]

            return x * y
            # # for i in range(c):
            # #    s.append(x*y)
            # # st =  tf.keras.backend.stack(s)
            # r = tf.keras.backend.reshape(st, shape=(-1,self.img_h,self.img_w,c))
            # return r
        radix_values = []
        for r in range(radix):
            radix_feature = radix_branch[r]
            radix_attention = tf.keras.layers.Lambda((lambda x:x[:,r]),output_shape=(1,filters))(attention)

            rv = tf.keras.layers.Lambda(elemwise_multiply,name="multiply{}-{}-r{}".format(layer,block,r),output_shape=(self.img_h,self.img_h,filters))([radix_feature,radix_attention,filters])            
            radix_values.append(rv)

        if radix == 1:
            feature = radix_values[0]
        else:
            feature = tf.keras.layers.Add()(radix_values)

        return feature

    def rsoftmax(self,x:tf.keras.layers.Layer, radix:int, filters:int):
        """[summary]

        Arguments:
            x {tf.keras.layers.Layer} -- [description]
            radix {int} -- [description]
            cardinality {int} -- [description]

        Returns:
            [type] -- [description]
        """
        if radix > 1:
            # x = tf.keras.layers.Reshape((radix,(fc2_cnn_channels//cardinality)))(x)
            # x = tf.keras.layers.Permute((2,1))(x)
            x = tf.keras.layers.Softmax(axis=1)(x)
            # x = x.reshape((0, self.cardinality, self.radix, -1)).swapaxes(1, 2)
            # x = x.softmax(x, axis=1)
            # x = x.reshape((0, -1))
        else:
            # not tested !!
            x = tf.keras.activation.Sigmoid(x)
        return x

    def update_img_size(self,stride):
        self.img_h = self.img_h // stride
        self.img_w = self.img_w // stride

    def build_resnet_base(self, input_shape:Tuple[int,int,int], size:int, mish:bool=True, resnet_c:bool=False, resnet_d:bool=False, resnest:bool=False)->tf.keras.Model:
        """
        mish: Mishの利用
        resnet_c: ResNet-C / ResNetのStemをDeep化する。
        resnet_d: ResNet-D / 残差のDownsampleをする際にAvgPoolingする。
        resnest : ブロック構築にSplit Attentionを使用する。

        """
        self.size = size

        self.img_h , self.img_w , _ = input_shape # 画像サイズを保存、以降、stride2で処理するごとに半分に。

        if mish:
            self.activation = "Mish"
        else:
            self.activation = "relu"

        if size not in [18,34,50,101,152]:
            raise ValueError('Wrong ResNet Size')

        # model = ResNet(Bottleneck, [3, 4, 6, 3],
        #               radix=2, cardinality=1, bottleneck_width=64,
        #               deep_stem=True,
        #               avg_down=True,
        #               avd=True, 
        #               avd_first=False,
        #               use_splat=True,  dropblock_prob=0.1,
        #               name_prefix='resnest_', **kwargs)
        # last_gamma = False

            # self.layer1 = self._make_layer(1,-stage_index,
            #                               block,
            #                               planes 64,
            #                               layers layers[0], 
            #                               avg_down=avg_down,
            #                               norm_layer=norm_layer,
            #                               last_gamma=last_gamma,
            #                               use_splat=use_splat　= True,
            #                               avd=avd = True)

                # def _make_layer(self, s
                #           tage_index,
                #            block,
                #            planes,
                #            blocks,
                #            strides=1,
                #            dilation=1,
                #            pre_dilation=1,
                #            avg_down=False,
                #            norm_layer=None,
                #            last_gamma=False,
                #           　dropblock_prob=0,
                #            input_size=224, 
                #           　use_splat=False,
                #            avd=False):


        self.resnet_d = resnet_d

        repeatations_table ={
            18: [1,1,1,1], # 18: [2,2,2,2],
            34: [3,4,6,3],
            50: [3,6,4,3],
            101: [3,6,23,3],
            152: [3,8,36,3],
        }
        repeatations = repeatations_table[size]

        initial_kernel_size=(7, 7)
        initial_strides=(2, 2)
        initial_pooling='max'
        initial_filters=64

        kr = tf.keras.regularizers.l2(1e-11)
        ki = 'he_normal'

        img_input = tf.keras.Input(shape=input_shape, name='input')

        # Conv1
        if resnet_c:
            # ResNet-C = deep stem
            # http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
            initial_kernel_size = (3,3)
            x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=initial_strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same",name=self.get_layer_name(1,0,"conv2d"))(img_input)
            x = tf.keras.layers.BatchNormalization(name=self.get_layer_name(1,0,"barchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(1,0,self.activation))(x)
            strides = (1,1)
            x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same",name=self.get_layer_name(1,1,"conv2d"))(img_input)
            self.update_img_size(2)
            x = tf.keras.layers.BatchNormalization(name=self.get_layer_name(1,1,"barchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(1,1,self.activation))(x)
            x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same",name=self.get_layer_name(1,2,"conv2d"))(img_input)
            x = tf.keras.layers.BatchNormalization(name=self.get_layer_name(1,2,"barchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(1,2,self.activation))(x)
            if resnest == False:
                x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same")(x)
                self.update_img_size(2)
        else:
            # 通常のResNetの入口=Stem
            x = tf.keras.layers.Conv2D(filters=initial_filters,kernel_size=initial_kernel_size,strides=initial_strides, kernel_initializer=ki, kernel_regularizer=kr,padding="same",name=self.get_layer_name(1,0,"conv2d"))(img_input)
            self.update_img_size(2)
            x = tf.keras.layers.BatchNormalization(name=self.get_layer_name(1,0,"barchnorm"))(x)
            x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(1,0,self.activation))(x)
            # 次のレイヤーはConv2層だが、Stemとして扱われる。
            if resnest == False:
                x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="same")(x)
                self.update_img_size(2)

        # Conv2-5
        filters = initial_filters

        for idx,r in enumerate(repeatations):
            if size <= 34:
                if resnest:
                    x = self.split_attention_simplified_block(input=x,layer=(idx + 2), filters=filters)
                else:
                    x = self.basic_block(x, layer=(idx + 2), filters=filters)
            else:
                if resnest:
                    x = self.split_attention_block(input=x,layer=(idx + 2), filters=filters)
                else:
                    x = self.bottleneck_block(x, layer=(idx + 2), filters=filters)

            filters = filters*2
    
        # final activation
        x = tf.keras.layers.BatchNormalization(axis=ResNetBuilder.CHANNEL_AXIS, name=self.get_layer_name(6,0,"final-batchnorm"))(x)
        x = tf.keras.layers.Activation(self.activation, name=self.get_layer_name(6,0,"final-"+self.activation))(x)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)

        model = tf.keras.Model(img_input, x)
        # model.summary()
        return model

class ResNet18(AbstractClassificationModel):

    def __init__(self):
        pass
    
    @classmethod
    def get_base_model(cls,input:(int,int,int),include_top:bool,weights:str,**kwargs)->tf.keras.Model:
        if include_top == True:
            raise NotImplementedError("This model not support include top")
        if weights is not None:
            raise NotImplementedError("This model not support include top")

        mish = kwargs.get('mish', False)
        resnet_c = kwargs.get("resnet_c", False)
        resnet_d = kwargs.get("resnet_d", False)
        resnest = kwargs.get("resnest", False)
        builder = ResNetBuilder()
        return builder.build_resnet_base(input,18, mish, resnet_c=resnet_c, resnet_d=resnet_d,resnest=resnest)



class ResNet34(AbstractClassificationModel):

    def __init__(self):
        pass

    @classmethod
    def get_base_model(cls,input:(int,int,int),include_top:bool,weights:str,**kwargs)->tf.keras.Model:
        if include_top == True:
            raise NotImplementedError("This model not support include top")
        if weights is not None:
            raise NotImplementedError("This model not support include top")

        mish = kwargs.get('mish', True)
        resnet_c = kwargs.get("resnet_c", False)
        resnet_d = kwargs.get("resnet_d", False)
        resnest = kwargs.get("resnest", False)
        builder = ResNetBuilder()
        return builder.build_resnet_base(input,34, mish, resnet_c=resnet_c, resnet_d=resnet_d,resnest=resnest)



class ResNet50(AbstractClassificationModel):

    def __init__(self):
        pass

    @classmethod
    def get_base_model(cls,input:(int,int,int),include_top:bool,weights:str,**kwargs)->tf.keras.Model:
        if include_top == True:
            raise NotImplementedError("This model not support include top")
        if weights is not None:
            raise NotImplementedError("This model not support include top")

        mish = kwargs.get('mish', True)
        resnet_c = kwargs.get("resnet_c", False)
        resnet_d = kwargs.get("resnet_d", False)
        resnest = kwargs.get("resnest", False)
        builder = ResNetBuilder()
        return builder.build_resnet_base(input,50, mish, resnet_c=resnet_c, resnet_d=resnet_d,resnest=resnest)





if __name__ == '__main__':

    builder = ResNetBuilder()
    model = builder.build_resnet_base(input_shape=(224,224,3),size=50)
