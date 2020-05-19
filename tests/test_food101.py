import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tftk
from tftk.image.dataset import Food1o1
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model import KerasResNet50V2
from tftk.train.image import ImageTrain
from tftk.image.augument import ImageAugument
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from tftk import Context

if __name__ == '__main__':


    context = Context.init_context(
        TRAINING_BASE_DIR="tmp",
        TRAINING_NAME="food101"
    )

    tftk.USE_MIXED_PRECISION()
    BATCH_SIZE = 64
    
    CLASS_NUM = 101
    IMAGE_SIZE = 224
    CHANNELS = 3
    EPOCHS = 100
    SHUFFLE_SIZE = 1000

    train, train_len = Food1o1.get_train_dataset()
    validation, validation_len = Food1o1.get_validation_dataset()

    train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageAugument.randaugment_map(1,2))
    train = train.map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.one_hot(CLASS_NUM),num_parallel_calls=tf.data.experimental.AUTOTUNE).apply(ImageAugument.mixup_apply(200,0.1))
    validation = validation.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.image_reguralization(),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(ImageDatasetUtil.one_hot(CLASS_NUM),num_parallel_calls=tf.data.experimental.AUTOTUNE)

    optimizer = OptimizerBuilder.get_optimizer(name="rmsprop", lr=0.05)
    model = KerasResNet50V2.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS),classes=CLASS_NUM) # resnest=True,resnet_c=True,resnet_d=True,mish=True)
    callbacks = CallbackBuilder.get_callbacks(tensorboard=False, consine_annealing=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=8)
    ImageTrain.train_image_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=EPOCHS)

