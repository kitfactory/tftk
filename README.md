# TFTK (TensorFlow Toolkit)

## 1. What is TFTK ?

TFTKはTensorFlowを簡単化するライブラリです。
まだ画像分類しかありませんが、学習するにあたって、あのモデルを使いたい、あのデータ拡張を使いたいなどを簡単にします。

* データセットの取得を簡単にします。
* データセットの操作を簡単にします。
* データの拡張を簡単にします。
* モデルも多数組み合わせられます。単純なCNNモデルから、ResNet/EfficientNetまで。


## インストール

pipコマンドでインストールすることができます。

> pip install tftk -U

## サンプル

__サンプル　mnistの学習__

に比較して、直接的なプログラミングを可能にします。
実質、半分以下の行数で記述できるでしょう。ほとんどのモデル、データセットが以下のプログラムの枠組みで記述できます。

__After__

```
from tftk.dataset.image import Mnist,ImageDatasetUtil
from tftk.image.model import SimpleClassificationModel
from tftk.image.callbacks import HandyCallback
from tftk.train.image import ImageTrain

# Mnistデータセットを取得し、9:1に分割します。
dataset, len = Mnist.get_train_dataset()
dataset = dataset.map(ImageDatasetUtil.dataset_init_classification(10))
(train, train_size),(validation, validation_size) = ImageDatasetUtil.devide_train_validation(dataset,len,0.9)

# 簡単なCNNモデルを用意します。
model = SimpleBaseModel.get_base_model(28,28,1)

# Optimizerやコールバックを取得します。
optimizer = Optimizer.get_optimizer() # SGD
callbacks = HandyCallback.get_callbacks() 

# 学習します
ImageTrain.train_image_classification(
    train_data=train,
    train_size = train_size,
    validation_data=validation,
    validation_size = validation_size,
    batch_size=50,
    shuffle_size=1000,
    model=model,
    callbacks=callbacks,
    optimizer=optimizer,
    loss="categorical_crossentropy"
)
```

ぜひ、以下と比べてください。

https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py



## 2.サポートされているパーツ

__データセット__

TensorFlow datasetsより、幾つかのデータセットをサポートします。


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
|Crop&Pad|あるサイズに対して、| |
|Crop&Pad|あるサイズに対して、| |

__データ拡張__

|データ化拡張|説明|リンク|
|:--|:--|:--|
|RandAugment|AutoAugument相当の精度向上をするSOTAデータ拡張手法です。| |
|Mixup|2つの画像を混ぜ合わせる画像拡張です。| |

__モデル__

|モデル|説明|リンク|
|:--|:--|:--|
|サンプルCNN|ごくごく簡単なCNNのモデルです。|--|
|ResNet18|小さな画像に好適なResNetの小規模なモデルです。| |
|ResNet34|小さな画像に好適なResNetの小規模なモデルです。| |
|ResNet50|ある程度の量の画像に好適なモデルです。| |
|ResNet50V2|ResNetの最も大きなモデルです。| |
|MobileNetV2|ResNetより軽量なモデルです。| |
|EfficientNet|(予定)|(予定)|

__活性化関数(Relu)__

以下の活性化関数でReluをオーバーライドすることが可能です。また、EfficientNetではswishからmishをデフォルトで使用します。

|活性化関数|説明|リンク|
|:--|:--|:--|
|Mish| | |

## オプション機能

##
 