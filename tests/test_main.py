import tensorflow as tf

from tftk.dataset.image.anomaly_detection import MVTecAd
from tftk.dataset.image.classification import Mnist
from tftk.dataset.image.classification import Cifar10
from tftk.dataset.image.classification import ImageLabelFolderDataset
from tftk.dataset.image.classification import Food1o1
from tftk.dataset.image.classification import PatchCamelyon
from tftk.model.image.base import SimpleBaseModel
from tftk.model.classify import SoftmaxClassifyModel
from tftk.dataset.image.utility import ImageDatasetUtil
from tftk.train.image import ImageTrain
from tftk.train.callbacks import HandyCallback

from PIL import Image

from tftk.augment.image import ImageAugument




# from tftk.model.image.base import ResNet50
# from tftk.model.image.base import MobileNetV2


if __name__ == '__main__':


    dataset, num = PatchCamelyon.get_train_dataset()
    

    """
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