from abc import ABC, abstractclassmethod
import tensorflow as tf

class BaseDataset():

    @abstractclassmethod
    def get_train_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        raise NotImplementedError()

    @abstractclassmethod
    def get_test_dataset(cls, **kwargs)->(tf.data.Dataset, int):
        raise NotImplementedError()
