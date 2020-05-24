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

    TRAINING_NO_EPOCH_LIMIT = "NO_EPOCH_LIMIT"
    TRAINING_BASE_DIR = "TRAINING_BASE_DIR"
    TRAINING_NAME = "TRAINING_NAME"

    def __init__(self):
        super(Context, self).__init__()
    
    @classmethod
    def init_context(cls, **kwargs):
        """コンテキストの初期化

        学習冒頭で与えられた引数でコンテキストを初期化する

        Example:
            Context.init_context(TRAINING_NAME="foo")

        """
        ctx = Context.get_instance()
        for k in kwargs.keys():
            ctx[k] = kwargs[k]
    
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
   
