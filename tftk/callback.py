from abc import ABC, abstractclassmethod

import tensorflow as tf
import math
import os

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


class CallbackBuilder():
    @classmethod
    def get_callbacks(cls, tensorboard_log_dir:str="./temp/log", save_weights="./tmp/model.hdf5", monitor="val_acc", consine_annealing=False, cosine_init_lr=0.01, cosine_max_epochs = 60):
        """よく利用するコールバックを設定します。
        
        Keyword Arguments:
            tensorboard_log_dir {str} -- tensorboardログを出力します。Noneの場合、出力しません。 (default: {None})
            save_weights {str} -- モデルを保存します。 (default: {"./tmp/model.hdf5"})
            monitor {str} -- [description] (default: {"val_acc"})
            max_epoch
        Returns:
            [type] -- [description]
        """

        callbacks = []
        if tensorboard_log_dir is not None:
            files = os.listdir(tensorboard_log_dir)
            num = -1
            for f in files:
                try:
                    tmp = int(f)
                    print("has tmp",tmp)
                    if tmp > num:
                        num = tmp
                except Exception as e:
                    pass
            if num != -1:
                num = num + 1
            tensorboard_log_dir = tensorboard_log_dir + os.path.sep + str(num)
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir))
        if save_weights is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_weights,monitor="val_acc",save_best_only=True,save_weights_only=True))
        if consine_annealing == True:
            cosine_annealer = CosineAnnealingScheduler(cosine_max_epochs,eta_max=cosine_init_lr,eta_min=0.0)
            callbacks.append(cosine_annealer)

        return callbacks

