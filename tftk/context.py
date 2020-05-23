"""学習時にコンテキストとして共有する情報を保持する。

"""

from typing import Dict

import os
from datetime import datetime

class Context(Dict):
    """tftk.Context
    
    コンテキストとして情報を保持する。
    コンテキストはDictとして利用できる。
    
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

        与えられた引数でコンテキストを初期化する

        Examples:

            引数はkwargsで渡す

            ::
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
   
