import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tftk
from tftk.image.dataset import Mnist
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model.classification import SimpleClassificationModel
from tftk.image.train import Trainer
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder

if __name__ == '__main__':

    BATCH_SIZE = 100
    CLASS_NUM = 10
    IMAGE_SIZE = 28
    EPOCHS = 2
    SHUFFLE_SIZE = 1000

    train, train_len = Mnist.get_train_dataset()
    validation, validation_len = Mnist.get_test_dataset()

    train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1),classes=CLASS_NUM)

    callbacks = CallbackBuilder.get_callbacks(base_dir = "tmp" , tensorboard=True, profile_batch="50,60" ,save_weights=True, consine_annealing=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)
    
    Trainer.train_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="binary_crossentropy",max_epoch=EPOCHS)

