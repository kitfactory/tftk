import tensorflow as tf


from PIL import Image

# from tftk.augment.image import ImageAugument

# from tftk.image.dataset import Mnist
# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import Food1o1
# from tftk.image.dataset import ImageLabelFolderDataset

# from tftk.image.dataset import ImageNetResized
# from tftk.image.dataset import PatchCamelyon
# from tftk.image.dataset import ImageCrawler

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tftk
from tftk.image.dataset import Mnist
from tftk.image.dataset import ImageDatasetUtil
from tftk.image.augument import ImageAugument
from tftk.image.dataset import ImageLabelFolderDataset

if __name__ == '__main__':

    BATCH_SIZE = 100
    CLASS_NUM = 10
    IMAGE_SIZE = 28
    EPOCHS = 2
    SHUFFLE_SIZE = 1000

    train, train_len = ImageLabelFolderDataset.get_train_dataset(name="dogs-vs-cats", manual_dir="tmp")

    train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(224,224))

    for d in train.take(1):
        image = d["image"]
        autocontrast_image = ImageAugument.autocontrast(image)
        array = autocontrast_image.numpy()
        
        # equalize_image = ImageAugument.equalize(image)
        # array = equalize_image.numpy()

        # invert_image = ImageAugument.invert(image)
        # array = invert_image.numpy()

        # rotate_image = ImageAugument.rotate(image,45)
        # array = rotate_image.numpy()

        # posterize_image = ImageAugument.posterize(image,6)
        # array = posterize_image.numpy()

        # brigtness_image = ImageAugument.brightness(image, 0.3)
        # array = brightness.numpy()

        # contrast_image = ImageAugument.contrast(image, 0.2)
        # array = contrast.numpy()

        # colored_image = ImageAugument.color(image, 0.01)
        # array = colored_image.numpy()

        # solarize_image = ImageAugument.solarize(image)
        # array = solarize_image.numpy()

        # solarize_add_image = ImageAugument.solarize_add(image)
        # array = solarize_add_image.numpy()

        # cutout_image = ImageAugument.cutout(image,100)
        #array = cutout_image.numpy()

        # translate_x_image = ImageAugument.translate_x(image,100)
        # translate_y_image = ImageAugument.translate_y(image,100)

        # sharpness_image = ImageAugument.sharpness(image, 0.95)
        # shear_x_image = ImageAugument.shear_x(image,0.5)
        im = Image.fromarray(array)
        im.show()
        # dataset = dataset.map(ImageAugument.randaugment_map(2,2))

    
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
