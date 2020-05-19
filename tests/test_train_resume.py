from tftk.image.dataset import Mnist
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model.classification import SimpleClassificationModel
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from tftk import Context

from tftk.train.image import ImageTrain

from tftk import ENABLE_SUSPEND_RESUME_TRAIN, IS_SUSPEND_RESUME_TRAIN, ResumeExecutor

if __name__ == '__main__':

    context = Context.init_context(
        TRAINING_BASE_DIR="tmp",
        TRAINING_NAME="resume_test"
    )

    ENABLE_SUSPEND_RESUME_TRAIN()

    BATCH_SIZE = 500
    CLASS_NUM = 10
    IMAGE_SIZE = 28
    EPOCHS = 3
    SHUFFLE_SIZE = 1000
    BASEDIR = "./tmp"
    TRAIN_NAME = "resume_test"

    # if IS_SUSPEND_RESUME_TRAIN() == True and IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE()== True:
    

    train, train_len = Mnist.get_train_dataset()
    validation, validation_len = Mnist.get_test_dataset()

    train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1),classes=CLASS_NUM)

    callbacks = CallbackBuilder.get_callbacks(tensorboard=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)
    
    ImageTrain.train_image_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=EPOCHS)
