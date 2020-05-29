import tensorflow as tf

import tftk
from tftk.image.dataset import MVTecAd
from tftk.image.dataset import Cifar10,ImageDatasetUtil
from tftk.image.model import SimpleAutoEncoderModel
from tftk.optimizer import OptimizerBuilder
from tftk.callback import CallbackBuilder
from tftk.image.dataset import ImageDatasetUtil

from tftk.train.image import ImageTrain


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

tftk.ENABLE_MIXED_PRECISION()

IMAGE_SIZE = 32
EPOCHS= 10
BATCH_SIZE = 50

# mvtec_ad, len = MVTecAd.get_train_dataset("bottle")
# mvtec_ad = mvtec_ad.map(ImageDatasetUtil.resize(IMAGE_SIZE,IMAGE_SIZE))
# (train, len),(validation, validation_len) =ImageDatasetUtil.devide_train_validation(mvtec_ad,len,0.9)

train, len = Cifar10.get_train_dataset()
validation, validation_len = Cifar10.get_train_dataset()
# validation,validation_len = MVTecAd.get_test_dataset("bottle")
# validation = validation.map(ImageDatasetUtil.resize(224,224))

train = train.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_r = validation.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE)

model = SimpleAutoEncoderModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))
optimizer = OptimizerBuilder.get_optimizer("rmsprop")
callback = CallbackBuilder.get_callbacks()
ImageTrain.train_image_autoencoder(train,len,BATCH_SIZE,validation_r,validation_len,100,model,callback,optimizer,"binary_crossentropy",EPOCHS,False)

ImageTrain.show_autoencoder_results(model,validation,20)

