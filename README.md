# TFTK (TensorFlow Toolkit)

## 1. What is TFTK ?

TFTKはTensorFlowを簡単化するライブラリです。
学習するにあたって、あのモデルを使いたい、あのデータ拡張を使いたい、あの何かをつかいたいを簡単にします。

* データセットの取得を簡単にします。
* データセットの操作を簡単にします。
* データの拡張を簡単にします。
* モデルも多数組み合わせられます。単純なCNNモデルから、ResNetまで。


## 2.サポートされているパーツ

__データセット操作__



__データ拡張__

|データ化拡張|説明|リンク|
|:--|:--|:--|
|Mixup| | |
|Cutout| | |
|RandAugment|| |

__モデル__

|モデル| | |
|:--|:--|:--|
|サンプルCNN| | |
|ResNet50| | |
|ResNet152| | |
|MobileNet| | |
|MobileNetV2| | |

__距離学習__

| | | |
|:--|:--|:--|
| | | |

__最適化__

| | | |
|:--|:--|:--|
|Mish| | |
|relu| | |

__CAM__




## サンプル

__サンプル　mnistの学習__

https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

に比較して、半分以下の労力で記述できます。

```
from tftk.dataset.image.classification import Mnist
from tftk.dataset.image.utility import ImageDatasetUtil
from tftk.model.image.base import SimpleBaseModel
from tftk.model.classify import SoftmaxClassifyModel
from tftk.train.callbacks import HandyCallback
from tftk.train.image import ImageTrain

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
