import os

import distutils
from distutils import dir_util

import tensorflow as tf

from . context import Context

COLAB_PATH = "/content/drive/My Drive"
# COLAB_PATH = "./tmp/colab"
COLAB_BACKUP_BASE_GDRIVE = COLAB_PATH + "/tftk/colab_back"

def IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE()->bool:
    return tf.io.gfile.exists(COLAB_PATH)

class Colaboratory():

    resumed = False

    @classmethod
    def copy_resume_data_from_google_drive(cls):
        context = Context.get_instance()
        name = context[Context.TRAINING_NAME]
        base = context[Context.TRAINING_BASE_DIR]

        colab = base + os.path.sep + name
        gdrive = COLAB_BACKUP_BASE_GDRIVE + os.path.sep + name

        if tf.io.gfile.exists(gdrive)==True and cls.resumed == False:
            print("copy files from gdrive")
            distutils.dir_util.copy_tree(gdrive, colab,verbose=1)
            cls.resume = True
        else:
            print("No resume data on gdrive.")

    @classmethod
    def copy_suspend_data_from_colab(cls):
        context = Context.get_instance()
        name = context[Context.TRAINING_NAME]
        base = context[Context.TRAINING_BASE_DIR]

        colab = base + os.path.sep + name
        gdrive = COLAB_BACKUP_BASE_GDRIVE + os.path.sep + name

        if tf.io.gfile.exists(gdrive) == False:
            print("google drive dirctory not found ", gdrive)
            os.makedirs(gdrive)
        else:
            print("gdrive found ", gdrive)

        print("\nbackup suspend data ", colab, gdrive,"\n")
        distutils.dir_util.copy_tree(colab,gdrive,verbose=1)

            
