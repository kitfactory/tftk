from typing import List
import time

import tensorflow as tf

from tftk.image.dataset import ImageDatasetUtil
from tftk import ResumeExecutor
from tftk import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE, Colaboratory
from PIL import Image
from tftk.image.dataset import ImageUtil

import matplotlib.pyplot as plt
import numpy as np
import cv2

class TrainingExecutor():
    """学習を提供するユーティリティです。

    """

    def __init__(self):
        pass
    
    @classmethod
    def train_classification(
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

        Return:
            history : 学習の結果

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
            Colaboratory.copy_resume_data_from_google_drive()
        else:
            print("google drive is not found.")

        if exe.is_resumable_training()==True:
            # print("This is resume training!!")
            exe.resume_model(model)
            resume_val = exe.resume_values()
            initial_epoch, _, _,_  = resume_val
            initial_epoch = initial_epoch + 1
            print("resuming epoch", initial_epoch, "max_epoch", max_epoch)
        else:
            if exe.is_train_ended()==True:
                print("Training is completed.")
                exit()
            else:
                # print("Not resume training")
                pass


        steps_per_epoch = train_size//batch_size
        validation_steps = validation_size//batch_size

        try:
            history = model.fit(
                train_data,
                callbacks=callbacks,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                epochs=max_epoch, initial_epoch=initial_epoch)
            return history
        except Exception as ex:
            raise ex
        finally:
            tf.keras.backend.clear_session()
            del optimizer,callbacks,model,train_data,validation_data
            if exe.is_resumable_training() == True:
                exe.training_completed()
            time.sleep(5)

    @classmethod
    def train_autoencoder(
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
        
        Return:
            history : 学習の結果

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
        try:
            history = model.fit(
                train_data,
                callbacks=callbacks,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                epochs=max_epoch, initial_epoch=initial_epoch)
            return history
        except Exception as ex:
            raise ex
        finally:
            tf.keras.backend.clear_session()
            del optimizer,callbacks,model,train_data,validation_data
            if exe.is_resumable_training() == True:
                exe.training_completed()
            time.sleep(5)

    @classmethod
    def show_autoencoder_results(cls, model:tf.keras.Model, dataset:tf.data.Dataset , size:int):
        """AutoEncoderで出力した値を表示

        """
     
        i = 0
        for data in dataset:
            img = data["image"]
            img = img.numpy()
            shape = img.shape
            h = shape[0]
            w = shape[1]
            c = shape[2]
            img = img.astype(np.uint8)

            if c == 1:
                img = np.reshape(img,[h,w])
                orig_im = Image.fromarray(img, 'L')
            else:
                orig_im = Image.fromarray(img)

            orig_im = orig_im.resize((256,256))
            x = np.reshape(img,[1,h,w,c])
            x = x / 255.0
            y = model.predict(x)
            y = y * 255
            y = y.astype(np.uint8)

            if c == 1:
                y = np.reshape(y,[h,w])
                y_im = Image.fromarray(y, 'L')
            else:
                y = np.reshape(y,[h,w,c])
                y_im = Image.fromarray(y)

            y_im = y_im.resize((256,256))

            if c == 1:
                concat = Image.new('L', (512,256))
            else:
                concat = Image.new('RGB',(512,256))
            concat.paste(orig_im,(0,0))
            concat.paste(y_im,(256,0))
            concat.show()

            if c != 1:
                diff = img - y
                diff = np.abs(diff)
                heatmap = np.mean(diff, axis=-1)
                heatmap = np.maximum(heatmap,0)
                heatmap /= np.max(heatmap)
                heatmap = np.uint8(255*heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                heatmap = ImageUtil.cv2pil(heatmap)
                heatmap.show()

            i += 1
            if i == size:
                return


    @classmethod
    def calucurate_reconstruction_error(cls,model:tf.keras.Model,  dataset:tf.data.Dataset, size:int):
        dataset = dataset.batch(1)
        i = 0
        for data in dataset:
            x = data["image"]
            x = x.numpy()
            m = x.shape[1]
            n = x.shape[2]
            mn = m * n
            i = i +1
            y = model.predict(x)
            loss = np.mean(tf.keras.losses.MSE(x,y), axis=(1,2)) / mn
            print(loss)
            if i > size:
                break

    
    @classmethod
    def train_gan(cls, generator:tf.keras.Model,  discriminator:tf.keras.Model, dataset:tf.data.Dataset, gan_input_shape=(32,), **kwargs):

        # discriminator コンパイル
        discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=0.008,clipvalue=1.0,decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')

        # 識別器はいったん学習させない
        discriminator.trainable = False

        # 
        gan_input = tf.keras.Input(shape=gan_input_shape)
        gan_output = discriminator(generator(gan_input))

        gan = tf.keras.Model(gan_input,gan_output)
        gan_optimizer = tf.keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,decay=1e-8)
        gan.compile()
