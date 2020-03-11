from abc import ABC, abstractclassmethod
import tensorflow as tf

class BaseDataset(ABC):
    """Datasetのベースクラス

    """

    @abstractclassmethod
    def get_train_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        """学習用のデータセットを取得する
        
        Arguments:
            cls -- クラスオブジェクト
        
        Returns:
            データセットとデータセットのサイズを持ったタプル
        """
        raise NotImplementedError()

    @abstractclassmethod
    def get_test_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        """テスト用のデータセットを取得する
        
        Arguments:
            cls -- クラスオブジェクト
        
        Returns:
            データセットとデータセットのサイズを持ったタプル
        """
        raise NotImplementedError()
