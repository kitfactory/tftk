from typing import Dict

class Context(Dict):

    instance = None

    MIXED_PRECISION = "MIXED_PRECISION"
    SUSPEND_RESUME = "SUSPEND_RESUME"

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
            cls.instance[Context.TRAINING_NAME] = "temp_name"
            cls.instance[Context.MIXED_PRECISION] = False
            cls.instance[Context.SUSPEND_RESUME] = False
        return cls.instance
   
