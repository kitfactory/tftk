import os
import tensorflow as tf
from . context import Context
from . colab import IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE, Colaboratory

def ENABLE_SUSPEND_RESUME_TRAIN():
    context = Context.get_instance()
    context[Context.SUSPEND_RESUME] = True

def IS_SUSPEND_RESUME_TRAIN():
    context = Context.get_instance()
    return context[Context.SUSPEND_RESUME]

class ResumeExecutor():

    instance = None

    RESUME_FILE = 'RESUME_INFO.txt'

    MODEL_FILE = 'model.h5'

    def __init__(self):
        context = Context.get_instance()
        self.base_dir = context[Context.TRAINING_BASE_DIR]
        self.name = context[Context.TRAINING_NAME]
        self.path = self.base_dir + os.path.sep + self.name

    @classmethod
    def get_instance(cls)->'ResumeExecutor':
        if cls.instance == None:
            cls.instance = ResumeExecutor()
        
        return cls.instance
    
    def _load_values(self)->(int,float,bool):
        f = tf.io.gfile.GFile(self._get_resume_path(),mode="r")
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
        context = Context.get_instance()
        if context[Context.SUSPEND_RESUME] == False:
            return False
        if tf.io.gfile.exists(self.path) == True:
            _, _, _, end =self._load_values()
            return end
        else:
            return False
    
    def is_resumable_training(self)->bool:
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
        lv = self._load_values()
        return lv

    def resume_model(self,model:tf.keras.Model):
        model.load_weights(self._get_model_path())

    def suspend(self,epoch,lr,best,model):
        print("\nSusupend!!!")
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
        self._check_dir()
        f = tf.io.gfile.GFile(self._get_resume_path(), mode="w")
        resume_info = f.write(str(0)+","+str(0.0)+"," +str(0.0)+","+ str(True))
        f.close()
        
        if IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE():
            Colaboratory.copy_suspend_data_from_colab()



