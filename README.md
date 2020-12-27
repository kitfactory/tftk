# TFTK (TensorFlow ToolKit)

## Downloads

Total : [![Downloads](https://pepy.tech/badge/tftk)](https://pepy.tech/project/tftk)

Weekly : [![Downloads](https://pepy.tech/badge/tftk/week)](https://pepy.tech/project/tftk/week)


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
from tftk.image.dataset import Mnist
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model.classification import SimpleClassificationModel
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from tftk import Context

from tftk.train.image import ImageTrain
from tftk import ENABLE_SUSPEND_RESUME_TRAINING, ResumeExecutor

BATCH_SIZE = 500
CLASS_NUM = 10
IMAGE_SIZE = 28
EPOCHS = 2
SHUFFLE_SIZE = 1000

# 学習コンテキストの準備、学習の保存設定
context = Context.init_context(TRAINING_NAME='mnist')
ENABLE_SUSPEND_RESUME_TRAINING()

# データセットの取得
train, train_len = Mnist.get_train_dataset()
validation, validation_len = Mnist.get_test_dataset()

# データセットの整形
train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))

# 最適化の準備
optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")

# モデルの準備
model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1),classes=CLASS_NUM)

# コールバック設定
callbacks = CallbackBuilder.get_callbacks(tensorboard=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)

# 学習の実施
ImageTrain.train_image_classification(
    train_data=train,train_size=train_len,
    batch_size=BATCH_SIZE,
    validation_data=validation,
    validation_size=validation_len,
    shuffle_size=SHUFFLE_SIZE,
    model=model,callbacks=callbacks,
    optimizer=optimizer,
    loss="categorical_crossentropy",
    max_epoch=EPOCHS)

```

ぜひ、以下と比べてください。
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py


## 2.サポートされているパーツ


### 2.1.データセット

TensorFlow datasetsより、幾つかのデータセットをサポートします。
get_train_dataset/get_test_datasetを行ってください。

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
|ラベルのone-hot化|ラベル値をone-hot形式にします。例：ラベル3->(0,0,1,0,0)|tftk.image.dataset.utility.ImageDatasetUtility#one_hote()|

### 2.3.データの操作

これらのメソッドは tf.data.Dataset#map()やapply()で利用できるように作られています。

|データ操作|説明|
|:--|:--|
|Resize|画像を学習するモデルに合わせて最適なサイズにします|
|Pad&Resize|正方形となるようパディングしながら画像をリサイズします。| 
|Crop&Resize|あるサイズに対して、最大となる正方形上に画像を切り出します|

### 2.4.データ拡張

これらのメソッドは tf.data.Dataset#map()やapply()で利用できるように作られています。

|データ化拡張|説明|リンク|
|:--|:--|:--|
|tf.image.augument.ImageAugument#randaugment_map()|AutoAugument相当の精度向上をするSOTAデータ拡張手法、RandAugumentです。| |
|tf.image.augment.ImageAugument#mixup_apply()|2つの画像をMixupする混ぜ合わせる画像拡張です。| |
|tf.image.augment.ImageAugument#cutout()|画像の一部を切り取るCutout拡張を行います| |

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









-------

Optunaとの融合

* 分散する
* MongoDB Storage

* シナリオ・最適化トレーニング

データ拡張のパラメータを追跡することが可能である。
さらに最適化とあわせて、コントロールすることが可能である。




--

## 特徴量の抽出
* 教師なしの場合、自分を学習対象にするオートエンコーダー
* 他の教師結果から、回復をするオートエンコーダー
* 滲みのない、特徴量の次元の少ないオートエンコーダーは何か。


## 
* アテンションの獲得

## 
* Grad-CAM
