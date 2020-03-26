import os
from os.path import expanduser

import tensorflow as tf
import tensorflow_datasets as tfds


class MVTecAd():
    """MVTec Adをデータセット

    ファイルを ~/dataset/mvtec_ad/(bottole etc..)に解凍します。
    
    """

    def __init__(self):
        """ 初期化
        """
        pass

    @classmethod
    def get_train_dataset(cls, type="bottle")->(tf.data.Dataset, int):
        manual_dir = expanduser("~") + os.path.sep + "dataset" + os.path.sep + "mvtec_ad"
        print(manual_dir)
        builder = tfds.image.ImageLabelFolder(type)

        dl_config = tfds.download.DownloadConfig(manual_dir=manual_dir)
        builder.download_and_prepare(download_config=dl_config)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        len = builder.info.splits['train'].num_examples  # Splits, num examples,... automatically extracted
        return ds, len

    @classmethod
    def get_test_dataset(cls, type="bottle")->(tf.data.Dataset,int):
        """test用データセットを取得する。
        
        Arguments:
            int {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        dir = expanduser("~") + os.path.sep + "dataset" + os.path.sep + "mvtec_ad" + os.path.sep + type + os.path.sep + "train" + os.path.sep + "good"
        print(dir)
        builder = tfds.image.ImageLabelFolder(dir)

        dl_config = tfds.download.DownloadConfig()
        builder.download_and_prepare(download_config=dl_config)
        print(builder.info)  # Splits, num examples,... automatically extracted
        ds, info = builder.as_dataset(split='test', shuffle_files=True, with_info=True)
        len = builder.info.splits['test'].num_examples
        return ds, len
    
