"""学習時にコンテキストとして共有する情報を保持する。

"""
from typing import Dict

import os
from datetime import datetime

class Context(Dict):
    """学習のコンテキスト保持クラスです。
    
    処理に必要な情報を保持します。
    コンテキストはDictとして利用できます。
    
    """
    instance = None

    MIXED_PRECISION = "MIXED_PRECISION"
    SUSPEND_RESUME = "SUSPEND_RESUME"
    OPTUNA = "OPTUNA"
    OPTUNA_TRIAL = "OPTUNA_TRIAL"

    TRAINING_NO_EPOCH_LIMIT = "NO_EPOCH_LIMIT"
    TRAINING_BASE_DIR = "TRAINING_BASE_DIR"
    TRAINING_NAME = "TRAINING_NAME"

    def __init__(self):
        super(Context, self).__init__()
    
    @classmethod
    def init_context(cls, training_name:str=None, **kwargs)->'Context':
        """コンテキストの初期化

        学習冒頭で与えられた引数でコンテキストを初期化する

        Example:
            Context.init_context(TRAINING_NAME="foo")

        """
        ctx = Context.get_instance()

        if training_name is not None:
            ctx[Context.TRAINING_NAME] = training_name

        for k in kwargs.keys():
            ctx[k] = kwargs[k]
        
        return ctx
    
    @classmethod
    def get_instance(cls)->'Context':
        """ Contextオブジェクトを取得する

        Returns:
            Context -- 生成されたコンテキスト
        """
        if cls.instance == None:
            cls.instance = Context()
            cls.instance[Context.TRAINING_BASE_DIR] ="tmp"
            date_dt = datetime.now()
            sdt = date_dt.strftime("%Y%m%d%H%M%S")
            cls.instance[Context.TRAINING_NAME] = sdt
            cls.instance[Context.MIXED_PRECISION] = False
            cls.instance[Context.SUSPEND_RESUME] = False
            cls.instance[Context.TRAINING_NO_EPOCH_LIMIT] = True
        return cls.instance   

    @classmethod
    def get_model_path(cls)->str:
        path = Context.get_training_path() + os.path.sep + 'model.h5'
        return path

    @classmethod
    def get_training_path(cls)->str:
        instance = Context.get_instance()
        path = instance[Context.TRAINING_BASE_DIR] + os.path.sep + instance[Context.TRAINING_NAME]
        if Context.OPTUNA_TRIAL in instance:
            path = path + os.path.sep + str(instance[Context.OPTUNA_TRIAL].number)
        return path

