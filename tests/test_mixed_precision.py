import tftk
from tftk.image.dataset import Place365Small
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model import ResNet50
from tftk.train import TrainingExecutor
from tftk.image.augument import ImageAugument
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder

BATCH_SIZE = 24
CLASS_NUM = 365
IMAGE_SIZE = 224
SHUFFLE_SIZE = 10000

# トレーニングデータ
train, train_len = Place365Small.get_train_dataset()
train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE))
train = train.map(ImageAugument.randaugment_map(3,10))
train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))

# バリデーションデータ
validation, validation_len = Place365Small.get_validation_dataset()
validation = validation.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE))
validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))

# Optimizerの準備
optimizer = OptimizerBuilder.get_optimizer(name="sgd",lr=0.01)
# モデルの準備
model = ResNet50.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),classes=CLASS_NUM)
# Callbackの準備
callbacks = CallbackBuilder.get_callbacks(tensorboard_log_dir="tmp\\log",save_weights="tmp\\weigths.hdf5", consine_annealing=False)

# トレーニングの実施
TrainingExecutor.train_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=50)