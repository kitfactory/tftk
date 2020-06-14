import os
import glob
from typing import List,Dict
import shutil

import tensorflow as tf
import tensorflow_datasets as tfds
from icrawler.builtin import BingImageCrawler

from tftk.image.dataset import BaseDataset

class Mnist(BaseDataset):
    @classmethod
    def get_train_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="mnist", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_validation_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        return (None, -1)


    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info =  tfds.load(name="mnist", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class Cifar10(BaseDataset):

    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="cifar10", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_validation_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        return (None, -1)

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="cifar10", split="test",with_info=True)
        return ds, info.splits["test"].num_examples



class ImageCrawler():

    @classmethod
    def crawl_keywords_save_folder(cls, name:str, keywords:List[str], base_dir:str="tmp", filters:Dict={"size":"large"}, max_num:int= 10000, train_ratio:float=0.8):
        """キーワードでクロールしてデータセットを作成する。
            Arguments:
            name {str} -- データセット名
            keywords {List[str]} -- キーワード
            filter {[type]} -- [description]

        Keyword Arguments:
            dest_base_dir {str} -- [description] (default: {"tmp/crawl"})
            train_ratio {float} -- [description] (default: {0.8})
        """
        download_base = base_dir + os.path.sep + name
        for k in keywords:
            download_dir = download_base + os.path.sep + "train" + os.path.sep + k
            if os.path.exists(download_dir) != True:
                os.makedirs(download_dir)
            storage={"root_dir": download_dir }

            print( "keyword:",k, " dir",download_dir)

            crawler = BingImageCrawler(storage=storage)
            crawler.crawl(keyword=k, filters=filters, max_num=max_num, file_idx_offset=0)


            move_dir = download_base + os.path.sep + "test" + os.path.sep + k
            if os.path.exists(move_dir) != True:
                os.makedirs(move_dir)

            file_list = glob.glob(download_dir+os.path.sep+"*.jpg")
            move_num:int = int(len(file_list) * train_ratio)

            move_list = file_list[move_num:]
            for f in move_list:
                shutil.move(f,move_dir)




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
    def get_validation_dataset(cls, name="dataset", dir="./", **kwargs)->(tf.data.Dataset,int):
        """validation用データセットを取得する。
        
        Arguments:
            int {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        builder = tfds.image.ImageLabelFolder(name)
        dl_config = tfds.download.DownloadConfig(manual_dir=dir)
        builder.download_and_prepare(download_config=dl_config)
        ds = builder.as_dataset(split='validation', shuffle_files=False)
        len = builder.info.splits['validation'].num_examples  # Splits, num examples,... automatically extracted
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


class Place365Small(BaseDataset):
    @classmethod
    def get_train_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="places365_small", split="train",with_info=True)
        return ds, info.splits["train"].num_examples
    
    @classmethod
    def get_validation_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="places365_small", split="validation",with_info=True)
        return ds, info.splits["validation"].num_examples

    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="places365_small", split="test",with_info=True)
        return ds, info.splits["test"].num_examples


class Food101(BaseDataset):
    @classmethod
    def get_train_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="food101", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_validation_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="food101", split="validation",with_info=True)
        return ds, info.splits["validation"].num_examples

    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        return (None,-1)

class PatchCamelyon(BaseDataset):
    @classmethod
    def get_train_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="patch_camelyon", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_validation_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        return (None,-1)

    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
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
    def get_train_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet_resized/64x64", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet_resized/64x64", split="validation",with_info=True)
        return ds, info.splits["validation"].num_examples

    @classmethod
    def get_test_dataset(cls,**kwargs)->(tf.data.Dataset, int):
        return (None,-1)


# bellows are working...

class ImageNet2012(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet2012", split="train",with_info=True,data_dir="D:\\imagenet")
        return ds, 1281167

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="imagenet2012", split="ttest",with_info=True,data_dir="D:\\imagenet")
        return ds, info.splits["test"].num_examples


class CatsVsDogs(BaseDataset):
    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="cats_vs_dogs", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        raise Exception("No test data")

class RockPaperScissors(BaseDataset):

    @classmethod
    def get_train_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="rock_paper_scissors", split="train",with_info=True)
        return ds, info.splits["train"].num_examples

    @classmethod
    def get_test_dataset(cls)->(tf.data.Dataset, int):
        ds, info = tfds.load(name="rock_paper_scissors", split="test",with_info=True)
        return ds, info.splits["test"].num_examples
