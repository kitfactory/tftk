"""学習時に使用するコールバックとユーティリティを提供する。

"""

from abc import ABC, abstractclassmethod

import tensorflow as tf
import math
import os
import sys
from tftk import ResumeExecutor
from tftk import Context
from tftk import IS_SUSPEND_RESUME_TRAINING
from tftk import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE
from tftk import Colaboratory

class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealingを行います。　以下のソースコードを参照(感謝) https://github.com/4uiiurz1/keras-cosine-annealing    
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

    tftk.ENABLE_SUSPEND_RESUME_TRAINING()を有効にすることで、自動的に利用されます。

    Example:
        import tftk
        from tftk.callbacks import HandyCallback
        from tftk.train.image import ImageTrain

        # 学習の有効化
        tftk.ENABLE_SUSPEND_RESUME_TRAINING()

        # コールバックの取得（ここで取得できるコールバックに自動追加されます)
        callback = HandyCallback.get_callbacks()

        ImageTrain.train_image_classification(xxxxxx, callback=callback)

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
        if IS_SUSPEND_RESUME_TRAINING() == True:
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
    def get_callbacks(cls, tensorboard:bool=True, profile_batch:str=None, consine_annealing=False, cosine_init_lr=0.01, cosine_max_epochs = 60, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.2,reduce_min=1e-6, early_stopping_patience=0, **kwargs):
        """よく利用するコールバックを簡単に取得できるようにします。

        デフォルトではTensorBoard,ReduceLROnPlateau(),EarlyStopping(val_loss非更新、10エポックで停止)が自動的に選ばれます。
        また、tftk.ENABLE_SUSPEND_RESUME_TRAINING()が有効な場合、中断/再開を可能にするSuspendCallbackが使用されます。
        
        Parameters:
            tensorboard : TensorBoardのログをBaseDir/TrainingName以下に保存する場合はTrueを指定する。未指定時 True
            cosine_annealing : CosineAnnealingを行い学習率をコントロールする場合はTrue、未指定時 False
            reduce_lr_on_plateau : ReduceLROnPlateauで停滞時に学習率を下げる場合はTrue 、未指定時 True
            ealy_stopping : EarlyStoppingで停滞時に学習を終了する場合、True。 未指定時 True.
            csv_logger : CSVLoggerを使用し、学習の記録を行う

        Keyword Arguments:
            profile_batch{str} -- プロファイルを行う際の開始バッチ、終了バッチを指定します。Noneの場合実行しません。
            monitor {str} -- [description] (default: {"val_acc"})
            annealing_epoch : コサイン・アニーリング全体のエポック数、指定なし 100エポック
            init_lr : コサイン・アニーリングする際の最初の学習率、未指定時 0.01
            min_lr : 最小の学習率、コサイン・アニーリング時 = 1e-6, ReduceOnPlateau時1e-6
            patience : ReduceOnPlateau使用時にpatienceエポック数、モニター値の更新がない場合、factor分学習率を下げる。
            early_stopping_patience : EalyStopping利用時に、このエポック数、monitor値の更新がなければ、学習を終了する。

        Returns:
            List[tf.keras.callbacks.Callback] -- 学習に使用するコールバックのList

        Example:
            from tftk.callbacks import HandyCallback
            callbacks = HandyCallback.get_callbacks(early_stopping_patience=15)
        """

        context = Context.get_instance()
        base = context[Context.TRAINING_BASE_DIR]
        name = context[Context.TRAINING_NAME]

        traing_dir = base + os.path.sep + name
        
        if tf.io.gfile.exists(traing_dir)== False:
            tf.io.gfile.makedirs(traing_dir)

        callbacks = []

        if tensorboard is True:
            # print("Callback-TensorBoard")
            tensorboard_log_dir = traing_dir + os.path.sep + "log"
            profile_batch = kwargs.get("profile_batch", None)
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
            annealing_epoch = kwargs.get("annealing_epoch", 100)
            init_lr = kwargs.get("init_lr", 0.01)
            min_lr = kwargs.get("min_lr", 1e-6)
            cosine_annealer = CosineAnnealingScheduler(annealing_epoch,eta_max=init_lr,eta_min=min_lr)
            callbacks.append(cosine_annealer)
            reduce_lr_on_plateau = False
        
        if reduce_lr_on_plateau ==True:
            print("Callback-ReduceOnPlateau")
            patience = kwargs.get("patience",5)
            factor = kwargs.get("factor",0.25)
            min_lr = kwargs.get("min_lr",1e-6)
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=patience,factor=factor,verbose=1,min_lr=min_lr))

        if early_stopping == True:
            early_stopping_patience = kwargs.get("early_stopping_patience", 8)
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience,verbose=1))

        if csv == True:
            callbacks.append(tf.keras.callbacks.CSVLogger());
        
        if IS_SUSPEND_RESUME_TRAINING() == True:
            print("Suspend Resume Callback")
            callbacks.append(SuspendCallback())

        return callbacks

