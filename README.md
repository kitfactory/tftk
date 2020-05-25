# TFTK (TensorFlow ToolKit)

## 1. What is TFTK ?

TFTKはTensorFlowを簡単化するライブラリです。
まだ画像分類しかできませんが、学習するにあたって、あのモデルを使いたい、あのデータ拡張を使いたいなどを簡単にします。

* データセットの取得を簡単にします。
* データセットの操作を簡単にします。
* データの拡張を簡単にします。
* モデルも多数組み合わせられます。単純なCNNモデルから、ResNet/EfficientNetまで。

## インストール

pipコマンドでインストールすることができます。

> pip install tftk -U

## サンプル

__サンプル　mnistの学習__

スクラッチに比較して、直接的なプログラミングを可能にします。ほとんどのモデル、データセットが以下のプログラムの枠組みで記述できます。おそらく、かなり短い行数になるでしょう。

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
optimizer = Optimizer.get_optimizer()
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


### 2.1.データセット

TensorFlow datasetsより、幾つかのデータセットをサポートします。

|データセット|クラス|説明|
|:--|:--|:--|
|Mnist|tftk.image.dataset.Mnist|Mnistです|
|Cifar10|tftk.image.dataset.Cifar10|Cifar10です|
|Food101|tftk.image.dataset.Food101|Food101です|
|ImageLabelFolderDataset|tftk.dataset.image.ImageLabelFolderDataset|フォルダごとに分けた画像をデータセットにします|

### 2.2. データセット操作

これらのメソッドは tf.data.Dataset#map()やapply()で利用できるように作られています。


|データ操作|説明|メソッド|
|:--|:--|:--|
|データセットの分割|データセットを指定の比率で分割します|tftk.image.dataset.utility.ImageDatasetUtility#devide_train_validation()|
|データセットの正規化 | 画素値を0-255から0.0～1.0に|tftk.image.dataset.utility.ImageDatasetUtility#image_reguralization()|
|ラベルのone-hot化|ラベル値をone-hot形式にします。ラベル3->(0,0,1,0,0)|tftk.image.dataset.utility.ImageDatasetUtility#one_hote()|

### 2.3.データの操作

これらのメソッドは tf.data.Dataset#map()やapply()で利用できるように作られています。

|データ操作|説明|
|:--|:--|:--|
|Resize|画像を学習するモデルに合わせて最適なサイズにします|
|Pad&Resize|正方形となるようパディングしながら画像をリサイズします。| 
|Crop&Resize|あるサイズに対して、最大となる正方形上に画像を切り出します|

### 2.4.データ拡張

これらのメソッドは tf.data.Dataset#map()やapply()で利用できるように作られています。

|データ化拡張|説明|リンク|
|:--|:--|:--|
|RandAugment|AutoAugument相当の精度向上をするSOTAデータ拡張手法です。| |
|Mixup|2つの画像を混ぜ合わせる画像拡張です。| |
|Cutout|画像の一部を切り取るCutout拡張を行います| |

## 2.5.モデル

|モデル|説明|リンク|
|:--|:--|:--|
|サンプルCNN|ごくごく簡単なCNNのモデルです。|tftk.image.model.SimpleClassificationModel|
|ResNet18|小さな画像に好適なResNetの小規模なモデルです。|tftk.image.model.KerasResNet18|
|ResNet34|小さな画像に好適なResNetの小規模なモデルです。|tftk.image.model.KerasResNet34|
|ResNet50|ある程度の量の画像に好適なモデルです。|tftk.image.model.KerasResNet50|
|ResNet50V2|ResNetの改良版です。|tftk.image.model.KerasResNet50V2|
|MobileNetV2|ResNetより軽量なモデルです。|tftk.image.model.KerasMobileNetV2|
|EfficientNet|最新の最強モデルです。まだTensorFlowにはいっていないので引っ張ってきたモデルです。|tftk.image.model.KerasEfficientNetBx|

## 2.6. 活性化関数

以下の活性化関数で、既存のモデルのReluをオーバーライドすることが可能です。また、EfficientNetモデルではswishではなく、mishをデフォルトで使用します。

|活性化関数|説明|リンク|
|:--|:--|:--|
|Mish|最強と思しき活性化関数|tftk.USE_MISH_AS_RELU()|

## 3.そのほかのオプション機能

使い方など順次、Qiitaに投稿するかもしれません。

* Google Colab用学習をGoogle Driveに退避/復帰 
* 混合精度(Mixed Precision)を使った学習

## 4.Special Thanks

本プロジェクトは多数のOSSで成り立っています。
ソースコード内にライセンスや参照先を記載しています。

