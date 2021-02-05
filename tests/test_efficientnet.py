import tensorflow as tf


"""
from tftk.dataset.image import Mnist
from tftk.dataset.image import Cifar10
from tftk.dataset.image import ImageLabelFolderDataset
from tftk.dataset.image import PatchCamelyon
from tftk.model.classify import SoftmaxClassifyModel
from tftk.dataset.image.utility import ImageDatasetUtil
from tftk.train.image import ImageTrain
from tftk.train.callbacks import HandyCallback
"""

from PIL import Image


# from tftk.augment.image import ImageAugument

# from tftk.image.dataset import Mnist
# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import Food1o1
# from tftk.image.dataset import ImageLabelFolderDataset

# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import PatchCamelyon
# from tftk.image.dataset import ImageCrawler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tftk
from tftk.image.dataset import Food101
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model import KerasEfficientNetB2

from tftk.train.image import ImageTrain
from tftk.image.augument import ImageAugument
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from typing import Dict


if __name__ == '__main__':


    tftk.ENABLE_MIXED_PRECISION()
    # tftk.ENABLE_SUSPEND_RESUME_TRAINING()

    BATCH_SIZE = 40

    CLASS_NUM = 101
    IMAGE_SIZE = 224
    CHANNELS = 3
    EPOCHS = 150
    SHUFFLE_SIZE = 1000

    train, train_len = Food101.get_train_dataset()
    validation, validation_len = Food101.get_validation_dataset()

    train = train.map(ImageDatasetUtil.map_max_square_crop_and_resize(IMAGE_SIZE,IMAGE_SIZE),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageAugument.randaugment_map(1,4))
    train = train.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.one_hot(CLASS_NUM),num_parallel_calls=tf.data.experimental.AUTOTUNE).apply(ImageAugument.mixup_apply(200,0.1))
    validation = validation.map(ImageDatasetUtil.map_max_square_crop_and_resize(IMAGE_SIZE,IMAGE_SIZE),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.one_hot(CLASS_NUM),num_parallel_calls=tf.data.experimental.AUTOTUNE)

    optimizer = OptimizerBuilder.get_optimizer()
    model = KerasEfficientNetB2.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS),classes=CLASS_NUM,weights="imagenet") # resnest=True,resnet_c=True,resnet_d=True,mish=True)
    callbacks = CallbackBuilder.get_callbacks(tensorboard=True, consine_annealing=False, reduce_lr_on_plateau=True,reduce_patience=6,reduce_factor=0.25,early_stopping_patience=10)
    ImageTrain.train_image_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=EPOCHS)

