"""学習時に使用するコールバックとユーティリティを提供する。

"""

from abc import ABC, abstractclassmethod

import tensorflow as tf
import math
import os
import sys
from tftk import ResumeExecutor
from tftk import Context
from tftk import IS_SUSPEND_RESUME_TRAIN
from tftk import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE
from tftk import Colaboratory

class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    Cosine annealingを行います。このソースコードは以下を参照しています。
    https://github.com/4uiiurz1/keras-cosine-annealing
    
    """

    def __init__(self, T_max:int, eta_max:float, eta_min:float=0, verbose=0):
        """初期化をします。
        
        Arguments:
            T_max {int} -- 最大エポック数
            eta_max {float} -- 最大学習率
        
        Keyword Arguments:
            eta_min {float} -- 最小学習率 (default: {0})
            verbose {int} -- バーバスモード(default: {0})
        """
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch:int, logs=None):
        """エポック開始前の処理を実施します。
        
        Arguments:
            epoch {int} -- エポック数
        
        Keyword Arguments:
            logs {[type]} -- [description] (default: {None})
        
        Raises:
            ValueError: [description]
        """
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        """エポック終了時の処理を実施します。
        
        Arguments:
            epoch {int} -- エポック数
        
        Keyword Arguments:
            logs {[type]} -- [description] (default: {None})
        
        Raises:
            ValueError: [description]
        """
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)



class SuspendCallback(tf.keras.callbacks.Callback):
    """学習の経過を保存し、次回実行時に再開させる情報を記録する

    Args:
        tf ([type]): [description]
    """

    def __init__(self,  suspend_on_best:bool =True , monitor:str="val_loss",path:str=None):
        self.suspend_on_best = suspend_on_best
        self.monitor = monitor
        self.path = path
        if monitor == "val_loss" or monitor == "loss":
            self.best_value = sys.float_info.max
        else:
            self.best_value = sys.float_info.min
        
        super(SuspendCallback, self).__init__()


    def on_train_begin(self, epoch:int, logs=None):
        """学習の開始処理

        Arguments:
            epoch {int} -- [description]

        Keyword Arguments:
            logs {[type]} -- [description] (default: {None})
        """
        if IS_SUSPEND_RESUME_TRAIN() == True:
            exe = ResumeExecutor.get_instance()
            if exe.is_resumable_training():
                resume = exe.resume_values()
                e,l,b,_ = resume
                self.best_value = b
                self.model.optimizer.lr = l
                # print("Suspend Call Resume LR",l)

    def on_epoch_begin(self, epoch:int, logs=None):
        pass

    def set_model(self, model):
        self.model:tf.kearas.Model = model
            
    def on_epoch_end(self, epoch, logs=None):
        if logs == None:
            return
        
        value = logs[self.monitor]
        lr = logs["lr"]
        
        exe = ResumeExecutor.get_instance()
        if self.monitor == "val_loss" or self.monitor == "loss":
            # print("\n monitor val_loss",self.best_value,",",value)
            if self.best_value > value:
                self.best_value = value
                exe.suspend(epoch,lr,self.best_value,self.model)
        else:
            if self.best_value < value:
                self.best_value = value
                exe.suspend(epoch,lr,self.best_value,self.model)
            
    def on_train_end(self, epoch:int, logs=None):
        context = Context.get_instance()
        if context[Context.TRAINING_NO_EPOCH_LIMIT] == False:
            exe = ResumeExecutor.get_instance()
            exe.training_completed()


class CallbackBuilder():
    @classmethod
    def get_callbacks(cls, tensorboard:bool=True, profile_batch:str=None, consine_annealing=False, cosine_init_lr=0.01, cosine_max_epochs = 60, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.2,reduce_min=1e-6, early_stopping_patience=0):
        """よく利用するコールバックを設定します。
        
        Parameters:
            base_dir: ログを行うベースディレクトリ
            name: トレーニングの名前、base_dir/nameにデータを保存する
            resume: Resumeの有無

        Keyword Arguments:
            tensorboard_log_dir {str} -- tensorboardログを出力します。Noneの場合、出力しません。 (default: {None})
            profile_batch{str} -- プロファイルを行う際の開始バッチ、終了バッチを指定します。Noneの場合実行しません。
            save_weights {str} -- モデルを保存します。 (default: {"./tmp/model.hdf5"})
            monitor {str} -- [description] (default: {"val_acc"})
            max_epoch

        Returns:
            [type] -- [description]
        """

        context = Context.get_instance()
        base = context[Context.TRAINING_BASE_DIR]
        name = context[Context.TRAINING_NAME]

        traing_dir = base + os.path.sep + name

        # if name != None:
        #     base = base_dir + os.path.sep + name
        # else:
        #     files = os.listdir(base_dir)
        #     max_num = 0
        #     for f in files:
        #         try:
        #             tmp = int(f)
        #             if tmp +1 > max_num:
        #                 max_num = tmp + 1
        #         except Exception as e:
        #             pass
        #     base = base_dir + os.path.sep + str(max_num)
        
        if tf.io.gfile.exists(traing_dir)== False:
            tf.io.gfile.makedirs(traing_dir)

        callbacks = []

        if tensorboard is True:
            # print("Callback-TensorBoard")
            tensorboard_log_dir = traing_dir + os.path.sep + "log"
            if profile_batch != None:
                callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,profile_batch=profile_batch,histogram_freq=1))
            else:
                callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir))

        # if save_weights == True:
        #     print("Callback-ModelCheckPoint")
        #     save_path = base + os.path.sep  + "model.hdf5"
        #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_path,monitor="val_acc",save_best_only=True,save_weights_only=True))
        if consine_annealing == True:
            print("Callback-CosineAnnealing")
            cosine_annealer = CosineAnnealingScheduler(cosine_max_epochs,eta_max=cosine_init_lr,eta_min=0.0)
            callbacks.append(cosine_annealer)
        if reduce_lr_on_plateau ==True:
            print("Callback-ReduceOnPlateau")
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=reduce_patience,factor=reduce_factor,verbose=1,min_lr=reduce_min))
        if early_stopping_patience != 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience,verbose=1))

        if IS_SUSPEND_RESUME_TRAIN() == True:
            print("Suspend Resume Callback")
            callbacks.append(SuspendCallback())

        return callbacks

