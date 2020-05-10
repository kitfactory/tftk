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

    train = train.map(ImageDatasetUtil.resize_with_crop_or_pad(224,224)).map(ImageDatasetUtil.one_hot(2))
    train = train.map(ImageDatasetUtil.image_reguralization()).apply(ImageAugument.mixup_apply(mixup_size=100, alpha=0.2))

    for d in train.take(1):
        image = d["image"] * 255
        image = tf.cast(image, tf.uint8)
        print(image)
        image = image.numpy()
        y = d["label"]
        print("y",d["label"].numpy())
        im = Image.fromarray(image)
        im.show()
