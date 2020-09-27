import tensorflow as tf
import tftk


def ssim_color_loss(y_true, y_pred):
    """ SSIM類似度を使用したカラー画像用のロス関数

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = tf.reduce_mean(tf.square((y_pred - y_true)),axis=[1,2,3])
    
    gray_y_true = tf.image.rgb_to_grayscale(y_true)
    gray_y_pred = tf.image.rgb_to_grayscale(y_pred)
    loss = 1 - tf.reduce_mean(tf.image.ssim(gray_y_true, gray_y_pred, 1.0)) + (5.0 *mse)
    if tftk.IS_MIXED_PRECISION():
        loss = tf.cast(loss, tf.float16)
    return loss
