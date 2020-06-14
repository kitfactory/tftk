import tensorflow as tf

def ssim_image_loss(y_true, y_pred):
    print(y_true, y_pred)
    y_true_mono = tf.keras.backend.mean(y_true, axis=3) # モノクロ化
    y_true_mono = tf.reshape(y_true_mono,[-1,128,128,1])
    y_pred_mono = tf.keras.backend.mean(y_true, axis=3) # モノクロ化
    y_pred_mono = tf.reshape(y_pred_mono,[-1,128,128,1])
    print(y_true_mono, y_pred_mono)

    return tf.reduce_mean(tf.image.ssim_multiscale(y_true_mono, y_pred_mono, 2.0))    
    # loss = tf.image.ssim(y_true_mono, y_pred_mono, max_val=1.0)
    # loss = tf.reshape(loss, [-1,1])
    # print(loss)
    # return loss

    # ut = tf.keras.backend.mean(y_true_mean, axis=(1,2))
    # st = tf.keras.backend.std(y_true_mean, axis=(1,2))
    # up = tf.keras.backend.mean(y_pred_mean, axis=(1,2))
    # sp = tf.keras.backend.std(y_true_mean, axis=(1,2))

    # k1=0.01
    # k2=0.03

    # print(ut,st)
    # ssim = (2.0 * ut * up) +  k1) (2*st*sp)

    loss2 = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred / 1000.0)))
    return loss2
