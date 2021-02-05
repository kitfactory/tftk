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

from tftk.train.image import ImageTrain
from tftk import Context
from tftk.loss import ssim_color_loss

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


train = train.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE) # .map(ImageDatasetUtil.resize(64,64))
validation_r = validation.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE) # .map(ImageDatasetUtil.resize(64,64))
model = SSIMAutoEncoderModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))

optimizer = OptimizerBuilder.get_optimizer("rmsprop")
callback = CallbackBuilder.get_callbacks()

loss = ssim_color_loss
ImageTrain.train_image_autoencoder(train,len,BATCH_SIZE,validation_r,validation_len,100,model,callback,optimizer,loss,EPOCHS,False)

# model.load_weights(Context.get_model_path())
ImageTrain.show_autoencoder_results(model,validation,15)
ImageTrain.calucurate_reconstruction_error(model,validation,10)

