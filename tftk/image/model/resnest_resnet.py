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
# norm_layer = BatchNorm

        


def bottleneck(channels,
    cardinality=1,
    bottleneck_width=64,
    strides=1,
    dilation=1,
    downsample=None,
    previous_dilation=1,
    norm_layer=None,
    norm_kwargs=None,
    last_gamma=False,
    dropblock_prob=0,
    input_size=None,
    use_splat=False,
    radix=2,
    avd=False,
    avd_first=False,
    in_channels=None,
    split_drop_ratio=0,
    **kwargs):
    expansion = 4
    group_width = int(channels * (bottleneck_width / 64.)) * cardinality
    norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
    dropblock_prob = dropblock_prob
    use_splat = use_splat
    avd = avd and (strides > 1 or previous_dilation != dilation)
    self.avd_first = avd_first


# ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ## Created by: Hang Zhang
# ## Email: zhanghang0704@gmail.com
# ## Copyright (c) 2020
# ##
# ## LICENSE file in the root directory of this source tree 
# ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# """ResNets, implemented in Gluon."""
# # pylint: disable=arguments-differ,unused-argument,missing-docstring
# from __future__ import division

# import os
# import math
# from mxnet.context import cpu
# from mxnet.gluon.block import HybridBlock
# from mxnet.gluon import nn
# from mxnet.gluon.nn import BatchNorm

# from .dropblock import DropBlock
# from .splat import SplitAttentionConv

# __all__ = ['ResNet', 'Bottleneck']

# def _update_input_size(input_size, stride):
#     sh, sw = (stride, stride) if isinstance(stride, int) else stride
#     ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
#     oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#     input_size = (oh, ow)
#     return input_size

# class Bottleneck(HybridBlock):
#     """ResNet Bottleneck
#     """
#     # pylint: disable=unused-argument
#     expansion = 4
#     def __init__(self, channels, cardinality=1, bottleneck_width=64, strides=1, dilation=1,
#                  downsample=None, previous_dilation=1, norm_layer=None,
#                  norm_kwargs=None, last_gamma=False,
#                  dropblock_prob=0, input_size=None, use_splat=False,
#                  radix=2, avd=False, avd_first=False, in_channels=None, 
#                  split_drop_ratio=0, **kwargs):
#         super(Bottleneck, self).__init__()
#         group_width = int(channels * (bottleneck_width / 64.)) * cardinality
#         norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
#         self.dropblock_prob = dropblock_prob
#         self.use_splat = use_splat
#         self.avd = avd and (strides > 1 or previous_dilation != dilation)
#         self.avd_first = avd_first
#         if self.dropblock_prob > 0:
#             self.dropblock1 = DropBlock(dropblock_prob, 3, group_width, *input_size)
#             if self.avd:
#                 if avd_first:
#                     input_size = _update_input_size(input_size, strides)
#                 self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
#                 if not avd_first:
#                     input_size = _update_input_size(input_size, strides)
#             else:
#                 input_size = _update_input_size(input_size, strides)
#                 self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
#             self.dropblock3 = DropBlock(dropblock_prob, 3, channels*4, *input_size)
#         self.conv1 = nn.Conv2D(channels=group_width, kernel_size=1,
#                                use_bias=False, in_channels=in_channels)
#         self.bn1 = norm_layer(in_channels=group_width, **norm_kwargs)
#         self.relu1 = nn.Activation('relu')
#         if self.use_splat:
#             self.conv2 = SplitAttentionConv(channels=group_width, kernel_size=3, strides = 1 if self.avd else strides,
#                                               padding=dilation, dilation=dilation, groups=cardinality, use_bias=False,
#                                               in_channels=group_width, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
#                                               radix=radix, drop_ratio=split_drop_ratio, **kwargs)
#         else:
#             self.conv2 = nn.Conv2D(channels=group_width, kernel_size=3, strides = 1 if self.avd else strides,
#                                    padding=dilation, dilation=dilation, groups=cardinality, use_bias=False,
#                                    in_channels=group_width, **kwargs)
#             self.bn2 = norm_layer(in_channels=group_width, **norm_kwargs)
#             self.relu2 = nn.Activation('relu')
#         self.conv3 = nn.Conv2D(channels=channels*4, kernel_size=1, use_bias=False, in_channels=group_width)
#         if not last_gamma:
#             self.bn3 = norm_layer(in_channels=channels*4, **norm_kwargs)
#         else:
#             self.bn3 = norm_layer(in_channels=channels*4, gamma_initializer='zeros',
#                                   **norm_kwargs)
#         if self.avd:
#             self.avd_layer = nn.AvgPool2D(3, strides, padding=1)
#         self.relu3 = nn.Activation('relu')
#         self.downsample = downsample
#         self.dilation = dilation
#         self.strides = strides

#     def hybrid_forward(self, F, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         if self.dropblock_prob > 0:
#             out = self.dropblock1(out)
#         out = self.relu1(out)

#         if self.avd and self.avd_first:
#             out = self.avd_layer(out)

#         if self.use_splat:
#             out = self.conv2(out)
#             if self.dropblock_prob > 0:
#                 out = self.dropblock2(out)
#         else:
#             out = self.conv2(out)
#             out = self.bn2(out)
#             if self.dropblock_prob > 0:
#                 out = self.dropblock2(out)
#             out = self.relu2(out)

#         if self.avd and not self.avd_first:
#             out = self.avd_layer(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         if self.dropblock_prob > 0:
#             out = self.dropblock3(out)

#         out = out + residual
#         out = self.relu3(out)

#         return out


# class ResNet(HybridBlock):
#     """ ResNet Variants Definations
#     Parameters
#     ----------
#     block : Block
#         Class for the residual block. Options are BasicBlockV1, BottleneckV1.
#     layers : list of int
#         Numbers of layers in each block
#     classes : int, default 1000
#         Number of classification classes.
#     dilated : bool, default False
#         Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
#         typically used in Semantic Segmentation.
#     norm_layer : object
#         Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
#         Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
#     last_gamma : bool, default False
#         Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
#     deep_stem : bool, default False
#         Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
#     avg_down : bool, default False
#         Whether to use average pooling for projection skip connection between stages/downsample.
#     final_drop : float, default 0.0
#         Dropout ratio before the final classification layer.
#     use_global_stats : bool, default False
#         Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
#         optionally set to True if finetuning using ImageNet classification pretrained models.
#     Reference:
#         - He, Kaiming, et al. "Deep residual learning for image recognition."
#         Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
#         - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
#     """
#     # pylint: disable=unused-variable
#     def __init__(self, block, layers, cardinality=1, bottleneck_width=64,
#                  classes=1000, dilated=False, dilation=1, norm_layer=BatchNorm,
#                  norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
#                  avg_down=False, final_drop=0.0, use_global_stats=False,
#                  name_prefix='', dropblock_prob=0, input_size=224,
#                  use_splat=False, radix=2, avd=False, avd_first=False, split_drop_ratio=0):
#         self.cardinality = cardinality
#         self.bottleneck_width = bottleneck_width
#         self.inplanes = stem_width*2 if deep_stem else 64
#         self.radix = radix
#         self.split_drop_ratio = split_drop_ratio
#         self.avd_first = avd_first
#         super(ResNet, self).__init__(prefix=name_prefix)
#         norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
#         if use_global_stats:
#             norm_kwargs['use_global_stats'] = True
#         self.norm_kwargs = norm_kwargs
#         with self.name_scope():
#             if not deep_stem:
#                 self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
#                                        padding=3, use_bias=False, in_channels=3)
#             else:
#                 self.conv1 = nn.HybridSequential(prefix='conv1')
#                 self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
#                                          padding=1, use_bias=False, in_channels=3))
#                 self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
#                 self.conv1.add(nn.Activation('relu'))
#                 self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
#                                          padding=1, use_bias=False, in_channels=stem_width))
#                 self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
#                 self.conv1.add(nn.Activation('relu'))
#                 self.conv1.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
#                                          padding=1, use_bias=False, in_channels=stem_width))
#             input_size = _update_input_size(input_size, 2)
#             self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
#                                   **norm_kwargs)
#             self.relu = nn.Activation('relu')
#             self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
#             input_size = _update_input_size(input_size, 2)
#             self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
#                                            norm_layer=norm_layer, last_gamma=last_gamma, use_splat=use_splat,
#                                            avd=avd)
#             self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
#                                            norm_layer=norm_layer, last_gamma=last_gamma, use_splat=use_splat,
#                                            avd=avd)
#             input_size = _update_input_size(input_size, 2)
#             if dilated or dilation==4:
#                 self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#                 self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4, pre_dilation=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#             elif dilation==3:
#                 # special
#                 self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#                 self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, dilation=2, pre_dilation=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#             elif dilation==2:
#                 self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#                 self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#             else:
#                 self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#                 input_size = _update_input_size(input_size, 2)
#                 self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
#                                                avg_down=avg_down, norm_layer=norm_layer,
#                                                last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                                input_size=input_size, use_splat=use_splat, avd=avd)
#                 input_size = _update_input_size(input_size, 2)
#             self.avgpool = nn.GlobalAvgPool2D()
#             self.flat = nn.Flatten()
#             self.drop = None
#             if final_drop > 0.0:
#                 self.drop = nn.Dropout(final_drop)
#             self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

#     def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
#                     pre_dilation=1, avg_down=False, norm_layer=None,
#                     last_gamma=False,
#                     dropblock_prob=0, input_size=224, use_splat=False, avd=False):
#         downsample = None
#         if strides != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
#             with downsample.name_scope():
#                 if avg_down:
#                     if pre_dilation == 1:
#                         downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
#                                                     ceil_mode=True, count_include_pad=False))
#                     elif strides==1:
#                         downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
#                                                     ceil_mode=True, count_include_pad=False))
#                     else:
#                         downsample.add(nn.AvgPool2D(pool_size=pre_dilation*strides, strides=strides, padding=1,
#                                                     ceil_mode=True, count_include_pad=False))
#                     downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
#                                              strides=1, use_bias=False, in_channels=self.inplanes))
#                     downsample.add(norm_layer(in_channels=planes * block.expansion,
#                                               **self.norm_kwargs))
#                 else:
#                     downsample.add(nn.Conv2D(channels=planes * block.expansion,
#                                              kernel_size=1, strides=strides, use_bias=False,
#                                              in_channels=self.inplanes))
#                     downsample.add(norm_layer(in_channels=planes * block.expansion,
#                                               **self.norm_kwargs))

#         layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
#         with layers.name_scope():
#             if dilation in (1, 2):
#                 layers.add(block(planes, cardinality=self.cardinality,
#                                  bottleneck_width=self.bottleneck_width,
#                                  strides=strides, dilation=pre_dilation,
#                                  downsample=downsample, previous_dilation=dilation,
#                                  norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
#                                  last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                  input_size=input_size, use_splat=use_splat, avd=avd, avd_first=self.avd_first,
#                                  radix=self.radix, in_channels=self.inplanes,
#                                  split_drop_ratio=self.split_drop_ratio))
#             elif dilation == 4:
#                 layers.add(block(planes, cardinality=self.cardinality,
#                                  bottleneck_width=self.bottleneck_width,
#                                  strides=strides, dilation=pre_dilation,
#                                  downsample=downsample, previous_dilation=dilation,
#                                  norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
#                                  last_gamma=last_gamma, dropblock_prob=dropblock_prob,
#                                  input_size=input_size, use_splat=use_splat, avd=avd, avd_first=self.avd_first,
#                                  radix=self.radix, in_channels=self.inplanes,
#                                  split_drop_ratio=self.split_drop_ratio))
#             else:
#                 raise RuntimeError("=> unknown dilation size: {}".format(dilation))

#             input_size = _update_input_size(input_size, strides)
#             self.inplanes = planes * block.expansion
#             for i in range(1, blocks):
#                 layers.add(block(planes, cardinality=self.cardinality,
#                                  bottleneck_width=self.bottleneck_width, dilation=dilation,
#                                  previous_dilation=dilation, norm_layer=norm_layer,
#                                  norm_kwargs=self.norm_kwargs, last_gamma=last_gamma,
#                                  dropblock_prob=dropblock_prob, input_size=input_size,
#                                  use_splat=use_splat, avd=avd, avd_first=self.avd_first,
#                                  radix=self.radix, in_channels=self.inplanes,
#                                  split_drop_ratio=self.split_drop_ratio))

#         return layers

#     def hybrid_forward(self, F, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = self.flat(x)
#         if self.drop is not None:
#             x = self.drop(x)
#         x = self.fc(x)

#         return x
