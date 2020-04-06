import tensorflow as tf
import tensorflow_datasets as tfds

from tftk.image.dataset import BaseDataset

class Mnist(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="mnist", split="train",with_info=True)
        return ds, info.splits["train"].num_examples
    
    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info =  tfds.load(name="mnist", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class Cifar10(BaseDataset):

    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="cifar10", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="cifar10", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class ImageLabelFolderDataset():

    @classmethod
    def get_train_dataset(cls, name="dataset", manual_dir="./", **kwargs)->(tf.data.Dataset, int):
        print(name,manual_dir)
        builder = tfds.image.ImageLabelFolder(name)
        dl_config = tfds.download.DownloadConfig(manual_dir=manual_dir)
        builder.download_and_prepare(download_config=dl_config)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        len = builder.info.splits['train'].num_examples  # Splits, num examples,... automatically extracted
        return ds, len

    @classmethod
    def get_test_dataset(cls, name="dataset", dir="./", **kwargs)->(tf.data.Dataset,int):
        """test用データセットを取得する。
        
        Arguments:
            int {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        builder = tfds.image.ImageLabelFolder(name)
        dl_config = tfds.download.DownloadConfig(manual_dir=dir)
        builder.download_and_prepare(download_config=dl_config)
        ds = builder.as_dataset(split='test', shuffle_files=False)
        len = builder.info.splits['test'].num_examples  # Splits, num examples,... automatically extracted
        return ds, len



class Food1o1(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="food101", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="food101", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class PatchCamelyon(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="patch_camelyon", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="patch_camelyon", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class ImageNetResized(BaseDataset):
    """ImageNetリサイズ画像
    
    Arguments:   
        BaseDataset {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet_resized/64x64", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet_resized/64x64", split="test",with_info=True)
        return ds, info.splits["test"].num_examples



class ImageNet2012(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet2012", split="train",with_info=True,data_dir="D:\\imagenet")
        return ds, 1281167

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet2012", split="ttest",with_info=True,data_dir="D:\\imagenet")
        return ds, info.splits["test"].num_examples
