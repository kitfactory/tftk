import tensorflow as tf

import tftk
from tftk.image.dataset import MVTecAd
from tftk.image.dataset import CatsVsDogs
from tftk.image.dataset import Mnist

from tftk.image.dataset import Cifar10,ImageDatasetUtil
from tftk.image.model import SimpleAutoEncoderModel
from tftk.image.model import SSIMAutoEncoderModel

from tftk.optimizer import OptimizerBuilder
from tftk.callback import CallbackBuilder
from tftk.image.dataset import ImageDatasetUtil

from tftk.loss import ssim_image_loss
from tftk.train.image import ImageTrain
from tftk import Context

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


tftk.Context.init_context(training_name='ssim_catsdog_deep_autoencoder')
tftk.ENABLE_MIXED_PRECISION()
tftk.ENABLE_SUSPEND_RESUME_TRAINING()

IMAGE_SIZE = 128
EPOCHS= 80
BATCH_SIZE = 50

# mvtec_ad, len = MVTecAd.get_train_dataset("bottle")
# mvtec_ad = mvtec_ad.map(ImageDatasetUtil.resize(IMAGE_SIZE,IMAGE_SIZE))
# (train, len),(validation, validation_len) =ImageDatasetUtil.devide_train_validation(mvtec_ad,len,0.9)


cats_vs_dogs, total_len = CatsVsDogs.get_train_dataset()
cats_vs_dogs = cats_vs_dogs.map(ImageDatasetUtil.map_max_square_crop_and_resize(IMAGE_SIZE,IMAGE_SIZE))
(train,len),(validation,validation_len) = ImageDatasetUtil.devide_train_validation(cats_vs_dogs,total_len,0.9)


# validation, validation_len = Cifar10.get_train_dataset()
# validation, validation_len = CatsVsDogs.get_test_dataset()
# validation = validation.map(ImageDatasetUtil.map_max_square_crop(IMAGE_SIZE,IMAGE_SIZE))

# validation,validation_len = MVTecAd.get_test_dataset("bottle")
# validation = validation.map(ImageDatasetUtil.resize(224,224))

# train, len = Cifar10.get_train_dataset()
# validation, validation_len = Cifar10.get_test_dataset()

train = train.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE) # .map(ImageDatasetUtil.resize(64,64))
validation_r = validation.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE) # .map(ImageDatasetUtil.resize(64,64))

# model = SimpleAutoEncoderModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))
model = SSIMAutoEncoderModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))

optimizer = OptimizerBuilder.get_optimizer("rmsprop")
callback = CallbackBuilder.get_callbacks()

def ssim_color_loss(y_true, y_pred):
  print(y_true,y_pred)

  mse = tf.reduce_mean(tf.square((y_pred - y_true)),axis=[1,2,3])

  gray_y_true = tf.image.rgb_to_grayscale(y_true)
  gray_y_pred = tf.image.rgb_to_grayscale(y_pred)
  return 1 - tf.reduce_mean(tf.image.ssim(gray_y_true, gray_y_pred, 1.0)) + (5.0 *mse)

  # s_y_true = tf.reduce_mean(y_true,axis=3)
  # s_y_pred = tf.reduce_mean(y_pred,axis=3)
  # return 1 - tf.reduce_mean(tf.image.ssim(s_y_true, s_y_pred, 1.0)) +  (3.0 * mse)

  return mse

  # r_y_true = tf.gather(y_true,[0],axis=3)
  # g_y_true = tf.gather(y_true,[1],axis=3)
  # b_y_true = tf.gather(y_true,[2],axis=3)

  # r_y_pred = tf.gather(y_pred,[0],axis=3)
  # g_y_pred = tf.gather(y_pred,[1],axis=3)
  # b_y_pred = tf.gather(y_pred,[2],axis=3)

  # return 1 - tf.reduce_mean(tf.image.ssim(r_y_true, r_y_pred, 1.0) + tf.image.ssim(g_y_true, g_y_pred, 1.0) + tf.image.ssim(b_y_true, b_y_pred , 1.0))


loss = ssim_color_loss

ImageTrain.train_image_autoencoder(train,len,BATCH_SIZE,validation_r,validation_len,100,model,callback,optimizer,loss,EPOCHS,False)

# model.load_weights(Context.get_model_path())
ImageTrain.show_autoencoder_results(model,validation,15)
ImageTrain.calucurate_reconstruction_error(model,validation,10)

