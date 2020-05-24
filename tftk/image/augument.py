import math

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from typing import Callable
from typing import Dict

class ImageAugument():
    """データ拡張処理を提供する。

    map_xxxxxなどは、tf.data.Dataset#map()で使用することのできる関数オブジェクトを提供する。
    applyのつく関数は、tf.data.Dataset#apply()で使用することのできる関数オブジェクトを提供する。

    一部のコードはTensorFlow tensorflow/tpu autoaugment.pyから参照されています。


    """

    _MAX_LEVEL = 10.

    @classmethod
    def mixup_apply(cls, mixup_size: int, alpha: float) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        """Mixup拡張をおこなう。ラベルはone-hot化されている事が必要です。

        Args:
            dataset (tf.data.Dataset): Mixupが適用されるデータセット
            mix_size (int): Mixupを行う前にシャッフルのバッファーサイズ
            alpha (float): Mixupのハイパーパラメータ (0.0-1.0)

        Returns:
            tf.data.Dataset: mixuped dataset.

        Example:
            from tftk.image.dataset import ImageDatasetUtil
            
            ...

            dataset:tf.data.Dataset = xxxx
            dataset.map(ImageDatasetUtil.one_hot())
            dataset = dataset.apply(ImageAugument.mixup_apply(100,0.1))

        """

        def mixup(dataset: tf.data.Dataset) -> tf.data.Dataset:
            def mixup_map(data, shuffled):
                dist = tfp.distributions.Beta([alpha], [alpha])
                beta = dist.sample([1])[0][0]

                ret = {}
                ret["image"] = (data["image"]) * beta + (shuffled["image"] * (1-beta))
                ret["label"] = (data["label"]) * beta + (shuffled["label"] * (1-beta))
                return ret

            shuffle_dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).shuffle(mixup_size)
            zipped = tf.data.Dataset.zip((dataset, shuffle_dataset))
            return zipped.map(mixup_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return mixup

    # from tensorflow/tpu autoaugment.py

    """
  Copyright 2017 The TensorFlow Authors.  All rights reserved.

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

   Copyright 2017, The TensorFlow Authors.

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

    @classmethod
    def _blend(cls, image1, image2, factor):
        """_blend image1 and image2 using 'factor'.
        Factor can be above 0.0.  A value of 0.0 means only image1 is used.
        A value of 1.0 means only image2 is used.  A value between 0.0 and
        1.0 means we linearly interpolate the pixel values between the two
        images.  A value greater than 1.0 "extrapolates" the difference
        between the two pixel values, and we clip the results to values
        between 0 and 255.

        Args:
            image1: An image Tensor of type uint8.
            image2: An image Tensor of type uint8.
            factor: A floating point value above 0.0.
        
        Returns:
            A _blended image Tensor of type uint8.
        """
        if factor == 0.0:
            return tf.convert_to_tensor(image1)
        if factor == 1.0:
            return tf.convert_to_tensor(image2)

        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = tf.cast(image1, tf.float32) + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
            return tf.cast(temp, tf.uint8)

        # Extrapolate:
        #
        # We need to clip and then cast.
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

    @classmethod
    def cutout_map(cls, pad_size):
        """cutoutのみ単体で拡張する場合のmap用関数
        Args:
            pad_size: カットするサイズ 縦横ともに(pad_size*2)pxを切り取り 
        Returns:
            カットアウト関数
        """

        def cutout(data):
            data["image"] = cls._cutout(data["image"], pad_size)
            return data

        return cutout

    @classmethod
    def cutout(cls,image, pad_size, replace=0):
        """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
        This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
        a random location within `img`. The pixel values filled in will be of the
        value `replace`. The located where the mask will be applied is randomly
        chosen uniformly over the whole image.
        Args:
            image: An image Tensor of type uint8.
            pad_size: Specifies how big the zero mask that will be generated is that
            is applied to the image. The mask will be of size
            (2*pad_size x 2*pad_size).
            replace: What pixel value to fill in the image in the area that has
            the cutout mask applied to it.
        Returns:
            An image Tensor that is of type uint8.
        """
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = tf.random.uniform(
            shape=[], minval=0, maxval=image_height,
            dtype=tf.int32)
        
        cutout_center_width = tf.random.uniform(
            shape=[], minval=0, maxval=image_width,
            dtype=tf.int32)

        lower_pad = tf.math.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.math.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.math.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.math.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(
            tf.zeros(cutout_shape, dtype=image.dtype),
            padding_dims, constant_values=1)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(
            tf.equal(mask, 0),
            tf.ones_like(image, dtype=image.dtype) * replace,
            image)
        return image


    @classmethod 
    def solarize(cls, image, threshold=128):
        # For each pixel in the image, select the pixel
        # if the value is less than the threshold.
        # Otherwise, subtract 255 from the pixel.
        return tf.where(image < threshold, image, 255 - image)

    @classmethod
    def solarize_add(cls, image, additionimage, addition=0, threshold=128):
        # For each pixel in the image less than threshold
        # we add 'addition' amount to it and then clip the
        # pixel value to be between 0 and 255. The value
        # of 'addition' is between -128 and 128.
        added_image = tf.cast(image, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        return tf.where(image < threshold, added_image, image)

    @classmethod
    def color(cls, image, factor):
        """Equivalent of PIL Color."""
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return cls._blend(degenerate, image, factor)
    
    @classmethod
    def contrast(cls, image, factor):
        """Equivalent of PIL Contrast."""
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # _blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return cls._blend(degenerate, image, factor)

    @classmethod
    def brightness(cls, image, factor):
        """Equivalent of PIL Brightness."""
        degenerate = tf.zeros_like(image)
        return cls._blend(degenerate, image, factor)

    @classmethod
    def posterize(cls, image, bits):
        """Equivalent of PIL Posterize."""
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    
    @classmethod
    def rotate(cls, image, degrees):
        """Rotates the image by degrees either clockwise or counterclockwise.
        Args:
            image: An image Tensor of type uint8.
            degrees: Float, a scalar angle in degrees to rotate all images by. If
            degrees is positive the image will be rotated clockwise otherwise it will
            be rotated counterclockwise.
        Returns:
            The rotated version of image.
        """
        # Convert from degrees to radians.
        print(image)
        degrees_to_radians = math.pi / 180.0
        radians = degrees * degrees_to_radians

        rotate_image = tfa.image.rotate(image, radians,name="rotate")
        # image = contrib_image.rotate(_warp(image), radians)
        return rotate_image

    @classmethod
    def translate_x(cls, image, pixels, replace=[128,128,128]):
        """Equivalent of PIL Translate in X dimension."""
        image = tfa.image.translate(image,[-pixels,0])
        # image = contrib_image.translate(_warp(image), [-pixels, 0])
        return image # un_warp(image, replace)

    @classmethod
    def translate_y(cls, image, pixels, replace=[128,128,128]):
        """Equivalent of PIL Translate in Y dimension."""
        image = tfa.image.translate(image,[0,-pixels])
        return image

    @classmethod
    def shear_x(cls, image, level, replace=[128,128,128]):
        """Equivalent of PIL Shearing in X dimension."""
        # Shear parallel to x axis is a projective transform
        # # with a matrix form of:
        # [1  level
        #  0  1].
        image = tfa.image.transform(
            image, [1., level, 0., 0., 1., 0., 0., 0.])
        return image

    @classmethod
    def shear_y(cls, image, level, replace=[128,128,128]):
        """Equivalent of PIL Shearing in Y dimension."""
        # Shear parallel to y axis is a projective transform
        # with a matrix form of:
        # [1  0
        #  level  1].
        image = tfa.image.transform(
            image, [1., 0., 0., level, 1., 0., 0., 0.])
        return image

    @classmethod
    def autocontrast(cls, image):
        """Implements Autocontrast function from PIL using TF ops.
            Args:
                image: A 3D uint8 tensor.
            Returns:
                The image after it has had autocontrast applied to it and will be of type
                uint8.
        """
        def scale_channel(image):
            """Scale the 2D image using the autocontrast rule."""
            # A possibly cheaper version can be done using cumsum/unique_with_counts
            # over the histogram values, rather than iterating over the entire image.
            # to compute mins and maxes.

            lo = tf.cast(tf.reduce_min(image),tf.float32)
            hi = tf.cast(tf.reduce_max(image),tf.float32)

            # Scale the image, making the lowest value 0 and the highest value 255.
            def scale_values(im):
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                im = tf.cast(im,tf.float32) * scale + offset
                im = tf.clip_by_value(im, 0.0, 255.0)
                return tf.cast(im, tf.uint8)

            result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
            return result

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image[:, :, 0])
        s2 = scale_channel(image[:, :, 1])
        s3 = scale_channel(image[:, :, 2])
        image = tf.stack([s1, s2, s3], 2)
        return image

    @classmethod
    def sharpness(cls, image, factor):
        """Implements Sharpness function from PIL using TF ops."""
        orig_image = image
        image = tf.cast(image, tf.float32)
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, 0)
        # SMOOTH PIL Kernel.
        kernel = tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
            shape=[3, 3, 1, 1]) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        degenerate = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding='VALID', dilations=[1, 1])
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

        # _blend the final result.
        return cls._blend(result, orig_image, factor)

    @classmethod
    def equalize(cls, image):
        """Implements Equalize function from PIL using TF ops."""
        def scale_channel(im, c):
            """Scale the data in the channel to implement equalize."""
            im = tf.cast(im[:, :, c], tf.int32)
            # Compute the histogram of the image channel.
            histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

            # For the purposes of computing the step, filter out the nonzeros.
            nonzero = tf.where(tf.not_equal(histo, 0))
            nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
            step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

            def build_lut(histo, step):
                # Compute the cumulative sum, shifting by step // 2
                # and then normalization by step.
                lut = (tf.cumsum(histo) + (step // 2)) // step
                # Shift lut, prepending with 0.
                lut = tf.concat([[0], lut[:-1]], 0)
                # Clip the counts to be in range.  This is done
                # in the C code for image.point.
                return tf.clip_by_value(lut, 0, 255)

            # If step is zero, return the original image.  Otherwise, build
            # lut from the full histogram and step and then index from it.
            result = tf.cond(tf.equal(step, 0),
                lambda: im,
                lambda: tf.gather(build_lut(histo, step), im))

            return tf.cast(result, tf.uint8)

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], 2)
        print(image)
        return image

    @classmethod
    def invert(cls, image):
        """Inverts the image pixels."""
        image = tf.convert_to_tensor(image)
        return 255 - image

    @classmethod
    def _randomly_negate_tensor(cls, tensor: tf.Tensor) -> tf.Tensor:
        """With 50% prob turn the tensor negative."""
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor

    @classmethod
    def _rotate_level_to_arg(cls, level):
        level = (level/cls._MAX_LEVEL) * 30.
        level = cls._randomly_negate_tensor(level)
        return (float(level),)

    @classmethod
    def _shrink_level_to_arg(cls, level):
        """Converts level to ratio by which we shrink the image content."""
        if level == 0:
            return (1.0,)  # if level is zero, do not shrink the image
        # Maximum shrinking ratio is 2.9.
        level = 2. / (cls._MAX_LEVEL / level) + 0.9
        return (level,)

    @classmethod
    def _enhance_level_to_arg(cls, level):
        return ((level/cls._MAX_LEVEL) * 1.8 + 0.1,)

    @classmethod
    def _shear_level_to_arg(cls, level):
        level = (level/cls._MAX_LEVEL) * 0.3
        # Flip level to negative with 50% chance.
        level = cls._randomly_negate_tensor(level)
        return (level,)

    @classmethod
    def _translate_level_to_arg(cls, level, translate_const):
        level = (level/cls._MAX_LEVEL) * float(translate_const)
        # Flip level to negative with 50% chance.
        level = cls._randomly_negate_tensor(level)
        return (level,)

    @classmethod
    def level_to_arg(cls, cutout_const, translate_const):
        return {
            'AutoContrast': lambda level: (), #0
            'Equalize': lambda level: (),
            'Invert': lambda level: (),
            'Rotate': cls._rotate_level_to_arg,
            'Posterize': lambda level: (int((level/cls._MAX_LEVEL) * 4),),
            'Solarize': lambda level: (int((level/cls._MAX_LEVEL) * 256),), #5
            'SolarizeAdd': lambda level: (int((level/cls._MAX_LEVEL) * 110),),
            'Color': cls._enhance_level_to_arg,
            'Contrast': cls._enhance_level_to_arg,
            'Brightness': cls._enhance_level_to_arg,
            'Sharpness': cls._enhance_level_to_arg, #10
            'ShearX': cls._shear_level_to_arg,
            'ShearY': cls._shear_level_to_arg,
            'Cutout': lambda level: (int((level/cls._MAX_LEVEL) * cutout_const),),
            # pylint:disable=g-long-lambda
            'TranslateX': lambda level: cls._translate_level_to_arg(
                level, translate_const),
            'TranslateY': lambda level: cls._translate_level_to_arg(
                level, translate_const),
            # pylint:enable=g-long-lambda
        }


    @classmethod
    def randaugment_map(cls, num:int, magnitude:int, cutout_const:int=40, translate_const:int=100):
        """ RandAugmentを行うmap関数を返却する。

            RandAugment is from the paper https://arxiv.org/abs/1909.13719,
            
            Args:
                num (int): 何種類の拡張を実施するか。大抵は[1,3]の範囲が良い値。論文中ではNとして記載。
                magnitude: 拡張の大きさ。全ての拡張操作共通で使用される。論文中では(M)で記載。
                　　　　　　　大抵、[5,30]の範囲にベストが存在。
            Returns:
                [type]: [description]
        """

        available_ops = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
            'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

        level = magnitude

        @tf.function
        def distort_each_ops(data):
            op_to_select = tf.random.uniform([], maxval=len(available_ops), dtype=tf.int32)
            # tf.print("ops ",op_to_select)
            ret = tf.switch_case(
                op_to_select,
                branch_fns={
                    0: lambda: cls.autocontrast(data["image"]),
                    1: lambda: cls.equalize(data["image"]),
                    2: lambda: cls.invert(data["image"]),
                    3: lambda: cls.rotate(data["image"], *(cls._rotate_level_to_arg(level))),
                    4: lambda: cls.posterize(data["image"] , int((level/cls._MAX_LEVEL)*4)),
                    5: lambda: cls.solarize(data["image"], int((level/cls._MAX_LEVEL)* 256)),
                    6: lambda: cls.solarize_add(data["image"],int((level/cls._MAX_LEVEL)* 110)),
                    7: lambda: cls.color(data["image"],*(cls._enhance_level_to_arg(level))),
                    8: lambda: cls.contrast(data["image"],*(cls._enhance_level_to_arg(level))),
                    9: lambda: cls.brightness(data["image"], *(cls._enhance_level_to_arg(level))),
                    10: lambda: cls.sharpness(data["image"], *(cls._enhance_level_to_arg(level))),
                    11: lambda: cls.shear_x(data["image"] , *(cls._shear_level_to_arg(level))),
                    12: lambda: cls.shear_y(data["image"] , *(cls._shear_level_to_arg(level))),
                    13: lambda: cls.translate_x(data["image"], *(cls._translate_level_to_arg(level, translate_const))),
                    14: lambda: cls.translate_y(data["image"], *(cls._translate_level_to_arg(level, translate_const))),
                    15: lambda: cls.cutout(data["image"], int((level/cls._MAX_LEVEL) * cutout_const)),
                },
                default= lambda: cls.autocontrast(data["image"], *())
            )
            return ret

        def distort_randaug(data):
            level = tf.cast(magnitude, tf.float32)
            # tf.print("level ", level)
            for i in range(num):
                image = distort_each_ops(data)
                data["image"] = image
            return data
        
        return distort_randaug

