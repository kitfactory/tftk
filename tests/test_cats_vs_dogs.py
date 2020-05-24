import tensorflow as tf
import tftk
from tftk import Context
from tftk.image.dataset import ImageLabelFolderDataset
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.augument import ImageAugument
from tftk.image.model import KerasResNet50
from tftk.image.model import KerasResNet18
from tftk.image.model import SimpleClassificationModel
from tftk.train.image import ImageTrain
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder

if __name__ == '__main__':

    CLASS_NUM = 10
    IMAGE_SIZE = 150
    IMAGE_CHANNELS = 3
    EPOCHS = 100
    BATCH_SIZE = 100

    tftk.ENABLE_MIXED_PRECISION()

    context = Context.init_context(TRAINING_NAME='DogsVsCats')
    train, train_len = ImageLabelFolderDataset.get_train_dataset(name="dogs-vs-cats", manual_dir="tmp")
    validation, validation_len = ImageLabelFolderDataset.get_validation_dataset(name="dogs-vs-cats", manual_dir="tmp")

    train = train.map(ImageDatasetUtil.map_max_square_crop_and_resize(IMAGE_SIZE,IMAGE_SIZE))
    train = train.map(ImageDatasetUtil.image_reguralization())
    train = train.map(ImageDatasetUtil.one_hot(CLASS_NUM))

    validation = validation.map(ImageDatasetUtil.map_max_square_crop_and_resize(IMAGE_SIZE,IMAGE_SIZE))
    validation = validation.map(ImageDatasetUtil.image_reguralization())
    validation = validation.map(ImageDatasetUtil.one_hot(CLASS_NUM))

    optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    model = KerasResNet18.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS),classes=CLASS_NUM)
    callbacks = CallbackBuilder.get_callbacks()
    ImageTrain.train_image_classification(
        train_data=train,train_size=train_len,
        batch_size=BATCH_SIZE,shuffle_size=100,
        validation_data=validation,validation_size=validation_len,
        model=model,callbacks=callbacks,
        optimizer=optimizer,loss="binary_crossentropy",max_epoch=EPOCHS)
