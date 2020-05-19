from typing import Dict

import os
from datetime import datetime

class Context(Dict):

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
        ctx = Context.get_instance()
        for k in kwargs.keys():
            ctx[k] = kwargs[k]
    
    @classmethod
    def get_instance(cls)->'Context':
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
   
