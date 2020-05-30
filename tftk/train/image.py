from typing import List
import tensorflow as tf

from tftk.image.dataset import ImageDatasetUtil
from tftk import ResumeExecutor
from tftk import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE, Colaboratory
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np


class ImageTrain():
    """画像を使用した学習を提供するユーティリティです。

    """

    def __init__(self):
        pass
    
    @classmethod
    def train_image_classification(
        cls,
        train_data:tf.data.Dataset, train_size:int, batch_size:int,
        validation_data:tf.data.Dataset, validation_size:int,
        shuffle_size:int,
        model:tf.keras.Model,
        callbacks:List[tf.keras.callbacks.Callback],
        optimizer:tf.keras.optimizers.Optimizer,
        loss:tf.keras.losses.Loss,
        max_epoch:int = 5, resume:bool = True):
        """画像分類の学習を実施します。
        
        Parameters:
            train_data{tf.data.Dataset}: 学習に使用するトレーニングデータ
            train_size{int}: トレーニングデータのデータ数
            batch_size{int} : 学習時のバッチサイズ
            shuffle_size : 学習時のデータシャッフルサイズ
            model{tf.keras.} : 学習モデル

        Example:
            import tftk


            tftk.Context.init_context(
                TRAINING_NAME = "example_traninig1"
                TRAINING_BASE_DIR = "./tmp"
            )
            tftk.ENABLE_SUSPEND_RESUME_TRAINING()
            tftk.USE_MIXED_PRECISION()
            

        """
        # dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

        train_data = train_data.map(ImageDatasetUtil.dict_to_classification_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
        if shuffle_size != 0:
            train_data = train_data.shuffle(shuffle_size)
        train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        validation_data = validation_data.map(ImageDatasetUtil.dict_to_classification_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_data = validation_data.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
        model.summary()

        initial_epoch = 0
        exe = ResumeExecutor.get_instance()

        if IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE():
            print("colab training with google drive")
            Colaboratory.copy_resume_data_from_google_drive()
        else:
            print("google drive is not found.")

        if exe.is_resumable_training()==True:
            print("This is resume training!!")
            exe.resume_model(model)
            resume_val = exe.resume_values()
            initial_epoch, _, _,_  = resume_val
            initial_epoch = initial_epoch + 1
            print("resume epoch", initial_epoch, "max_epoch", max_epoch)
        else:
            if exe.is_train_ended()==True:
                print("Training is completed.")
                exit()
            else:
                print("Not resume training")

        """
        optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
        sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
        distribute=None, **kwargs
        """
        steps_per_epoch = train_size//batch_size
        validation_steps = validation_size//batch_size
        model.fit(
            train_data,
            callbacks=callbacks,
            validation_data=validation_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=max_epoch, initial_epoch=initial_epoch)






    @classmethod
    def train_image_autoencoder(
        cls,
        train_data:tf.data.Dataset, train_size:int, batch_size:int,
        validation_data:tf.data.Dataset, validation_size:int,
        shuffle_size:int,
        model:tf.keras.Model,
        callbacks:List[tf.keras.callbacks.Callback],
        optimizer:tf.keras.optimizers.Optimizer,
        loss:tf.keras.losses.Loss,
        max_epoch:int = 5, resume:bool = True):
        """AutoEncoderの学習を実施します。
        
        Parameters:
            train_data{tf.data.Dataset}: 学習に使用するトレーニングデータ
            train_size{int}: トレーニングデータのデータ数
            batch_size{int} : 学習時のバッチサイズ
            shuffle_size : 学習時のデータシャッフルサイズ
            model{tf.keras.Model} : 学習モデル

        Example:
            import tftk


            tftk.Context.init_context(
                TRAINING_NAME = "example_traninig1"
                TRAINING_BASE_DIR = "./tmp"
            )
            tftk.ENABLE_SUSPEND_RESUME_TRAINING()
            tftk.USE_MIXED_PRECISION()
            
        """    

   
        train_data = train_data.map(ImageDatasetUtil.dict_to_autoencoder_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
        if shuffle_size != 0:
            train_data = train_data.shuffle(shuffle_size)
        train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        validation_data = validation_data.map(ImageDatasetUtil.dict_to_autoencoder_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_data = validation_data.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        model.compile(optimizer=optimizer, loss=loss)
        model.summary()

        initial_epoch = 0
        exe = ResumeExecutor.get_instance()

        if IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE():
            print("colab training with google drive")
            Colaboratory.copy_resume_data_from_google_drive()
        else:
            print("google drive is not found.")

        if exe.is_resumable_training()==True:
            print("This is resume training!!")
            exe.resume_model(model)
            resume_val = exe.resume_values()
            initial_epoch, _, _,_  = resume_val
            initial_epoch = initial_epoch + 1
            print("resume epoch", initial_epoch, "max_epoch", max_epoch)
        else:
            if exe.is_train_ended()==True:
                print("Training is completed.")
                exit()
            else:
                print("Not resume training")

        steps_per_epoch = train_size//batch_size
        validation_steps = validation_size//batch_size
        model.fit(
            train_data,
            callbacks=callbacks,
            validation_data=validation_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=max_epoch, initial_epoch=initial_epoch)



    @classmethod
    def show_autoencoder_results(cls, model:tf.keras.Model, dataset:tf.data.Dataset , size:int):
     
        i = 0
        for data in dataset:
            img = data["image"]
            img = img.numpy()
            shape = img.shape
            h = shape[0]
            w = shape[1]
            c = shape[2]
            orig_im = Image.fromarray(img.astype(np.uint8))
            orig_im = orig_im.resize((256,256))
            img = np.reshape(img,[1,h,w,c])
            x = img / 255.0
            y = model.predict(x)
            y = y * 255
            y = y.astype(np.uint8)
            y = np.reshape(img,[h,w,c])
            y_im = Image.fromarray(y)
            y_im = y_im.resize((256,256))
            y_im.show()

            concat = Image.new('RGB',(512,256))
            concat.paste(orig_im,(0,0))
            concat.paste(y_im,(256,0))
            plt.imshow(np.array(concat))
            i += 1
            if i == size:
                return


"""
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)
"""