# TFTK (TensorFlow Toolkit)

## 1. What is TFTK ?

TFTKはTensorFlowを簡単化するライブラリです。
まだ画像分類しかありませんが、学習するにあたって、あのモデルを使いたい、あのデータ拡張を使いたいなどを簡単にします。

* データセットの取得を簡単にします。
* データセットの操作を簡単にします。
* データの拡張を簡単にします。
* モデルも多数組み合わせられます。単純なCNNモデルから、ResNet/EfficientNetまで。

## 2.サポートされているパーツ

__データセット操作__

|データ操作|説明|メソッド|
|:--|:--|:--|
|データセットの分割|tftk.image.dataset.utility.ImageDatasetUtility#devide_train_validation()|
|データセットの正規化|画素値を0-255から0.0～1.0に|
|ラベルのone-hot化|tftk.image.dataset.utility.ImageDatasetUtility#one_hote()|

__データの操作__

|データ操作|説明|メソッド|
|:--|:--|:--|
|Resize|画像を学習するモデルに合わせて最適なサイズにします| |
|Crop&Pad|あるサイズに対して| |

__データ拡張__

|データ化拡張|説明|リンク|
|:--|:--|:--|
|Mixup|2つの画像を混ぜ合わせる画像拡張です。| |
|RandAugment|AutoAugument相当の精度向上をするSOTAデータ拡張手法です。| |

__モデル__

|モデル|説明|リンク|
|:--|:--|:--|
|サンプルCNN|ごくごく簡単なCNNのモデルです。|--|
|ResNet18|小さな画像に好適なResNetの小規模なモデルです。| |
|ResNet34|小さな画像に好適なResNetの小規模なモデルです。| |
|ResNet50|ある程度の量の画像に好適なモデルです。| |
|ResNet152|ResNetの最も大きなモデルです。| |
|MobileNet|ResNetより軽量なモデルです。| |
|MobileNetV2|ResNetより軽量なモデルです。| |
|EfficientNet|(予定)|(予定)|

__活性化関数(Relu)__

以下の活性化関数でReluをオーバーライドすることが可能です。

|活性化関数|説明|リンク|
|:--|:--|:--|
|Mish| | |

## サンプル

__サンプル　mnistの学習__

https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

に比較して、半分以下の労力で記述できます。

__After__

```
from tftk.image.dataset import Mnist,ImageDatasetUtil
from tftk.image.model import SimpleClassificationModel
from tftk.image.callbacks import HandyCallback
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

__Before__
