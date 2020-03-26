import math

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from typing import Callable
from typing import Dict

class ImageAugument():

    _MAX_LEVEL = 10.

    @classmethod
    def mixup_apply(cls, dataset: tf.data.Dataset, mixup_size: int, alpha: float) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        """Mixup拡張をおこなう。ラベルはone-hot化されている事。
           dataset:tf.data.Dataset = somedataset
           mixuped = dataset.apply(DataAugument.mixup_apply(100,0.8))

        Args:
            dataset (tf.data.Dataset): dataset to be applied mixup augmentation.
            mix_size (int): data shuffle size to mixup
            alpha (float): Hyper parameter of the mixup. (0-1)

        Returns:
            tf.data.Dataset: mixuped dataset.
        """

        def mixup(cls, dataset: tf.data.Dataset) -> tf.data.Dataset:
            shuffle_dataset = dataset.shuffle(mix_size)
            zipped = tf.data.Dataset.zip((dataset, shuffle_dataset))

            def __mixup_map(data, shuffled):
                print(data)
                print(shuffled)
                dist = tfp.distributions.Beta([alpha], [alpha])
                beta = dist.sample([1])[0][0]

                ret = {}
                ret["image"] = (data["image"]) * beta + \
                    (shuffled["image"] * (1-beta)),
                ret["label"] = (data["label"]) * beta + \
                    (shuffled["label"] * (1-beta)),
                return ret

            return zipped.map(__mixup_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return mixup

    # from tensorflow/tpu autoaugment.py

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
            data["image"] = cls._cutout(data["image"], padsize)
            return data

        return cutout

    @classmethod
    def _cutout(cls,image, pad_size, replace=0):
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
    def _solarize(cls, image, threshold=128):
        # For each pixel in the image, select the pixel
        # if the value is less than the threshold.
        # Otherwise, subtract 255 from the pixel.
        return tf.where(image < threshold, image, 255 - image)

    @classmethod
    def _solarize_add(cls, image, additionimage, addition=0, threshold=128):
        # For each pixel in the image less than threshold
        # we add 'addition' amount to it and then clip the
        # pixel value to be between 0 and 255. The value
        # of 'addition' is between -128 and 128.
        added_image = tf.cast(image, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        return tf.where(image < threshold, added_image, image)

    @classmethod
    def _color(cls, image, factor):
        """Equivalent of PIL Color."""
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return cls._blend(degenerate, image, factor)
    
    @classmethod
    def _contrast(cls, image, factor):
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
    def _brightness(cls, image, factor):
        """Equivalent of PIL Brightness."""
        degenerate = tf.zeros_like(image)
        return cls._blend(degenerate, image, factor)

    @classmethod
    def _posterize(cls, image, bits):
        """Equivalent of PIL Posterize."""
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    
    @classmethod
    def _rotate(cls, image, degrees):
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
        degrees_to_radians = math.pi / 180.0
        radians = degrees * degrees_to_radians

        # In practice, we should randomize the rotation degrees by flipping
        # it negatively half the time, but that's done on 'degrees' outside
        # of the function.
        rotate_image = tfa.image.rotate(image, radians)
        # image = contrib_image.rotate(_warp(image), radians)
        return rotate_image

    @classmethod
    def _translate_x(cls, image, pixels, replace=[128,128,128]):
        """Equivalent of PIL Translate in X dimension."""
        image = tfa.image.translate(image,[-pixels,0])
        # image = contrib_image.translate(_warp(image), [-pixels, 0])
        return image # un_warp(image, replace)

    @classmethod
    def _translate_y(cls, image, pixels, replace=[128,128,128]):
        """Equivalent of PIL Translate in Y dimension."""
        image = tfa.image.translate(image,[0,-pixels])
        return image

    @classmethod
    def _shear_x(cls, image, level, replace=[128,128,128]):
        """Equivalent of PIL Shearing in X dimension."""
        # Shear parallel to x axis is a projective transform
        # # with a matrix form of:
        # [1  level
        #  0  1].
        image = tfa.image.transform(
            image, [1., level, 0., 0., 1., 0., 0., 0.])
        return image

    @classmethod
    def _shear_y(cls, image, level, replace=[128,128,128]):
        """Equivalent of PIL Shearing in Y dimension."""
        # Shear parallel to y axis is a projective transform
        # with a matrix form of:
        # [1  0
        #  level  1].
        image = tfa.image.transform(
            image, [1., 0., 0., level, 1., 0., 0., 0.])
        return image

    @classmethod
    def _autocontrast(cls, image):
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
    def _sharpness(cls, image, factor):
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
    def _equalize(cls, image):
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
    def _invert(cls, image):
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
                    0: lambda: cls._autocontrast(data["image"]),
                    1: lambda: cls._equalize(data["image"]),
                    2: lambda: cls._invert(data["image"]),
                    3: lambda: cls._rotate(data["image"], *(cls._rotate_level_to_arg(level))),
                    4: lambda: cls._posterize(data["image"] , int((level/cls._MAX_LEVEL)*4)),
                    5: lambda: cls._solarize(data["image"], int((level/cls._MAX_LEVEL)* 256)),
                    6: lambda: cls._solarize_add(data["image"],int((level/cls._MAX_LEVEL)* 110)),
                    7: lambda: cls._color(data["image"],*(cls._enhance_level_to_arg(level))),
                    8: lambda: cls._contrast(data["image"],*(cls._enhance_level_to_arg(level))),
                    9: lambda: cls._brightness(data["image"], *(cls._enhance_level_to_arg(level))),
                    10: lambda: cls._sharpness(data["image"], *(cls._enhance_level_to_arg(level))),
                    11: lambda: cls._shear_x(data["image"] , *(cls._shear_level_to_arg(level))),
                    12: lambda: cls._shear_y(data["image"] , *(cls._shear_level_to_arg(level))),
                    13: lambda: cls._translate_x(data["image"], *(cls._translate_level_to_arg(level, translate_const))),
                    14: lambda: cls._translate_y(data["image"], *(cls._translate_level_to_arg(level, translate_const))),
                    15: lambda: cls._cutout(data["image"], int((level/cls._MAX_LEVEL) * cutout_const)),
                },
                default= lambda: cls._autocontrast(data["image"], *())
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

