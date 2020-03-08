# TFTK (TensorFlow Toolkit)

## What is TFTK ?

TFTKはTensorFlowを簡単化するライブラリです。

* データセットの取得を簡単にします。
* データセットの操作を簡単にします。
* モデルも多数組み合わせられます。
　単純なCNNモデルから、ResNetまで。

__サンプル　mnistの学習__
```
from tfkit.dataset.image.classification import Mnist
from tfkit.model.image.base import SimpleBaseModel
from tfkit.model.classify import SoftmaxClassifyModel
from tfkit.dataset.image.utility import ImageDatasetUtil
from tfkit.train.image import ImageTrain
from tfkit.train.callbacks import HandyCallback

dataset, len = Mnist.get_train_dataset()
dataset = dataset.map(ImageDatasetUtil.dataset_init_classification(10))
(train, train_size),(validation, validation_size) = ImageDatasetUtil.devide_train_validation(dataset,len,0.9)
model = SimpleBaseModel.get_base_model(28,28,1)
model = SoftmaxClassifyModel.get_classify_model(model,10)
callbacks = HandyCallback.get_callbacks()

ImageTrain.train_image_classification(
    train_data=train,
    train_size = train_size,
    validation_data=validation,
    validation_size = validation_size,
    batch_size=50,
    shuffle_size=1000,
    model=model,
    callbacks=callbacks,
    optimizer="rmsprop",
    loss="categorical_crossentropy"
)
```
