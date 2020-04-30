import tensorflow as tf



"""
from tftk.dataset.image import Mnist
from tftk.dataset.image import Cifar10
from tftk.dataset.image import ImageLabelFolderDataset
from tftk.dataset.image import Food1o1
from tftk.dataset.image import PatchCamelyon
from tftk.model.classify import SoftmaxClassifyModel
from tftk.dataset.image.utility import ImageDatasetUtil
from tftk.train.image import ImageTrain
from tftk.train.callbacks import HandyCallback
"""

from PIL import Image

# from tftk.augment.image import ImageAugument

# from tftk.image.dataset import Mnist
# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import Food1o1
# from tftk.image.dataset import ImageLabelFolderDataset

# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import PatchCamelyon
# from tftk.image.dataset import ImageCrawler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tftk
from tftk.image.dataset import Place365Small
from tftk.image.dataset import RockPaperScissors
from tftk.image.dataset import CatsVsDogs
from tftk.image.dataset import ImageDatasetUtil
# from tftk.image.model.classify import ResNet50
# from tftk.image.model.classify import MobileNetV2
from tftk.image.model.classification import SimpleClassificationModel
from tftk.image.dataset import ImageLabelFolderDataset
from tftk.image.model.small_resnet_keras_contrib import ResNet18
from tftk.image.train import Trainer
from tftk.image.augument import ImageAugument
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder

# from tftk.model.image.base import ResNet50
# from tftk.model.image.base import MobileNetV2

if __name__ == '__main__':

    # tftk.USE_MIXED_PRECISION()
    # tftk.USE_MISH_AS_RELU()
    # dataset, len = Food1o1.get_train_dataset()
    # ImageCrawler.crawl_keywords_save_folder(name="animals",keywords=["ライオン","ゾウ","キリン","ネコ","イヌ","コウテイペンギン","ジェンツーペンギン","ワラビー","チーター","ピューマ"], base_dir="tmp",max_num=3000,train_ratio=0.8)
    # ImageCrawler.crawl_keywords_save_folder(name="nintendo",keywords=["とたけけ","たぬきち","ゼルダ","マリオ","ピカチュウ","ヒトカゲ","ヨッシー","クラウド","ティファ"], base_dir="tmp",max_num=3000,train_ratio=0.8)
    #  (train,train_len),(validation,validation_len)=ImageDatasetUtil.devide_train_validation(dataset,len,0.9)

    # dataset, len = PatchCamelyon.get_train_dataset()
    # train,train_len = ImageLabelFolderDataset.get_train_dataset(name='animals',manual_dir='tmp')
    # validation, validation_len = ImageLabelFolderDataset.get_test_dataset(name='animals', manual_dir='tmp')


    tftk.USE_MIXED_PRECISION()
    BATCH_SIZE = 100

    # BATCH_SIZE = 48
    CLASS_NUM = 2
    IMAGE_SIZE = 150
    EPOCHS = 100

    train, train_len = ImageLabelFolderDataset.get_train_dataset(name="dogs-vs-cats", manual_dir="tmp")
    validation, validation_len = ImageLabelFolderDataset.get_validation_dataset(name="dogs-vs-cats", manual_dir="tmp")

    # dataset,dataset_len = CatsVsDogs.get_train_dataset()
    # dataset = dataset.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE))
    # (train, train_len), (validation, validation_len) = ImageDatasetUtil.devide_train_validation(dataset,dataset_len,0.90)
    train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE)).map(ImageAugument.randaugment_map(1,4))
    train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    validation = validation.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE)).map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    # optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    optimizer = OptimizerBuilder.get_optimizer(name="adabound")
    model = ResNet18.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),classes=CLASS_NUM)
    # model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),classes=CLASS_NUM)
    callbacks = CallbackBuilder.get_callbacks(base_dir = "tmp" , tensorboard=True,save_weights=True, consine_annealing=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)
    Trainer.train_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=10000,model=model,callbacks=callbacks,optimizer=optimizer,loss="binary_crossentropy",max_epoch=EPOCHS)
    
    """
    train, train_len = RockPaperScissors.get_train_dataset()
    validation, validation_len = Place365Small.get_validation_dataset()
    train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE))
    validation = validation.map(ImageDatasetUtil.resize_with_crop_or_pad(IMAGE_SIZE,IMAGE_SIZE))
    dataset, len  = MVTecAd.get_train_dataset(type="bottle")
    print(dataset)
    print(len)

    for d in dataset.take(1):
        image = d["image"]
        # autocontrast_image = ImageAugument.autocontrast(image)
        # equalize_image = ImageAugument.equalize(image)
        # invert_image = ImageAugument.invert(image)
        # rotate_image = ImageAugument.rotate(image,45)
        # posterize_image = ImageAugument.posterize(image,6)
        # solarize_image = ImageAugument.solarize(image)
        ##  solarize_add_image = ImageAugument.solarize_add(image)
        # colored_image = ImageAugument.color(image, 0.01)
        # contrast_image = ImageAugument.contrast(image, 0.2)
        # brigtness_image = ImageAugument.brightness(image, 0.3)
        # sharpness_image = ImageAugument.sharpness(image, 0.95)
        # shear_x_image = ImageAugument.shear_x(image,0.5)
        # translate_x_image = ImageAugument.translate_x(image,100)
        # translate_y_image = ImageAugument.translate_y(image,100)
        # cutout_image = ImageAugument.cutout(image,100)
        # array = cutout_image.numpy()
        # im = Image.fromarray(array)
        # im.show()
        dataset = dataset.map(ImageAugument.randaugment_map(2,2))

        for data in dataset.take(1):
            array = data["image"].numpy()
            im = Image.fromarray(array)
            im.show()
    """

    """

    dataset, len = Mnist.get_train_dataset()

    dataset = dataset.map(ImageDatasetUtil.dataset_init_classification(10)) # .map(ImageDatasetUtil.resize(50,50))
    (train, train_size),(validation, validation_size) = ImageDatasetUtil.devide_train_validation(dataset,len,0.9)
    
    model = SimpleBaseModel.get_base_model(28,28,1)
    model = SoftmaxClassifyModel.get_classify_model(model,10)
    callbacks = HandyCallback.get_callbacks()

    print(callbacks)

    ImageTrain.train_image_classification(
        train_data=train,
        train_size = train_size,
        validation_data=validation,
        validation_size = validation_size,
        batch_size=300,
        shuffle_size=1000,
        model=model,
        callbacks=callbacks,
        optimizer="rmsprop",
        loss="categorical_crossentropy"
    )
    """

    # train,train_size,50,validation,validation_size,1000,model,None,"rmsprop","categorical_crossentropy")
    # model:tf.kearas.Model = ConvAutoEncoder.get_model()
