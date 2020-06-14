from tftk.image.dataset import Mnist
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model.classification import SimpleClassificationModel
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from tftk import Context

from optuna import Trial,Study
from tftk.optuna import Optuna
from tftk.train import TrainingExecutor

from tftk import ENABLE_SUSPEND_RESUME_TRAINING, ResumeExecutor


BATCH_SIZE = 500
CLASS_NUM = 10
IMAGE_SIZE = 28
EPOCHS = 10
SHUFFLE_SIZE = 1000

def train(trial:Trial):
    context = Optuna.get_optuna_conext('minist_optuna', trial)
    print("New trial ", trial.number , "++++++++++++++++++++++++++++" , context)
    ENABLE_SUSPEND_RESUME_TRAINING()

    print(context)
    Optuna.suggest_float(name='lr',low=1e-6, high=1e-2,log=True)
    train, train_len = Mnist.get_train_dataset()
    validation, validation_len = Mnist.get_test_dataset()

    train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    optimizer = OptimizerBuilder.get_optimizer(name="rmsprop", lr=Optuna.get_value('lr', default=0.1))
    model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1),classes=CLASS_NUM)
    callbacks = CallbackBuilder.get_callbacks(tensorboard=True, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)
    history = TrainingExecutor.train_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=EPOCHS)

    return history.history['val_loss'][-1]

if __name__ == '__main__':
    Optuna.start_optuna(train, num_of_trials=10)

    # if IS_SUSPEND_RESUME_TRAIN() == True and IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE()== True:


