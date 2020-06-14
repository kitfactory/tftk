""" 学習の一時中断、再開を行う処理を提供する

"""

import os
import tensorflow as tf
from . context import Context
from . colab import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE, Colaboratory

def ENABLE_SUSPEND_RESUME_TRAINING():
    """ 一時中断、再開を行う学習を実施する

    冒頭でリジュームを宣言してください。
    
    Example:
        ENABLE_SUSPEND_RESUME_TRAINING()

        ... 通常の学習
        ...

    """
    context = Context.get_instance()
    context[Context.SUSPEND_RESUME] = True

def IS_SUSPEND_RESUME_TRAINING()->bool:
    """ 一時中断、再開を行う学習を実施するかを確認する
    
    Returns:
        bool : 中断、再開を行う場合、True

    """
    context = Context.get_instance()
    return context[Context.SUSPEND_RESUME]

class ResumeExecutor():
    """再開処理を提供する

    """

    instance = None

    RESUME_FILE = 'RESUME_INFO.txt'

    MODEL_FILE = 'model.h5'

    def __init__(self):
        self.path = Context.get_training_path()

    @classmethod
    def get_instance(cls)->'ResumeExecutor':
        """インスタンスを取得する

        Returns:
            ResumeExecutor -- [description]
        """
        if cls.instance == None:
            cls.instance = ResumeExecutor()
        
        return cls.instance
    
    def _load_values(self)->(int,float,bool):
        rp = self._get_resume_path()
        f = tf.io.gfile.GFile(rp,mode="r")
        s = f.read()
        splited = s.split(",")
        epoch = int(splited[0])
        lr = float(splited[1])
        best = float(splited[2])
        end = (splited[3]=="True")
        f.close()
        ret = (epoch,lr,best,end)
        # print(ret)
        return ret

    def is_train_ended(self)->bool:
        """再開用データを確認し、学習が終了しているかを返却する

        Returns:
            bool -- 再開用データが存在していて、終了している場合はTrue
        """
        context = Context.get_instance()
        if context[Context.SUSPEND_RESUME] == False:
            return False
        if tf.io.gfile.exists(self._get_model_path()) == True:
            _, _, _, end =self._load_values()
            return end
        else:
            return False
    
    def is_resumable_training(self)->bool:
        """ 再開可能な学習かを返却する

        Returns:
            bool -- 再開可能かどうか
        """
        context = Context.get_instance()
        if context[Context.SUSPEND_RESUME] == False:
            return False

        if tf.io.gfile.exists(self._get_model_path()):
            return not self.is_train_ended()
        else:
            return False

    def _check_dir(self):
        if tf.io.gfile.exists(self.path) == False:
            tf.io.gfile.makedirs(self.path)

    def _get_model_path(self)->str:
        return self.path + os.path.sep + ResumeExecutor.MODEL_FILE
    
    def _get_resume_path(self)->str:
        return self.path + os.path.sep + ResumeExecutor.RESUME_FILE

    def resume_values(self)->(int,float):
        """ 中断データをファイルから読み取る

        Returns:
            int : エポック数
            float : 学習率
        """
        lv = self._load_values()
        return lv

    def resume_model(self,model:tf.keras.Model):
        """モデルを再開する

        Arguments:
            tf.keras.Model -- 重みを読み込み、再開したいモデル（コンパイル済）
        """
        model.load_weights(self._get_model_path())

    def suspend(self,epoch,lr,best,model):
        """ 中断可能なデータを保存する

        Arguments:
            epoch {int} -- 保存するエポック
            lr {float} -- 保存する学習率
            best {float} -- モニター値
            model {tf.kera.Model} -- 保存するモデル
        """
        self._check_dir()
        file = self._get_resume_path()
        f = tf.io.gfile.GFile(file, mode="w")
        c = str(epoch)+","+str(lr)+"," + str(best) + "," +str(False)
        print("suspend ",file,c)
        resume_info = f.write(c)
        f.flush()
        f.close()
        if model != None:
            model.save_weights(self._get_model_path())
        
        if IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE():
            Colaboratory.copy_suspend_data_from_colab()

    def training_completed(self):
        """学習が完了したことを記録する

        """
        self._check_dir()
        f = tf.io.gfile.GFile(self._get_resume_path(), mode="w")
        resume_info = f.write(str(0)+","+str(0.0)+"," +str(0.0)+","+ str(True))
        f.close()
        
        if IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE():
            Colaboratory.copy_suspend_data_from_colab()
        
        # 終了時にクリアする
        ResumeExecutor.instance = None

    
