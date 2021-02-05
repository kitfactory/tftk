
import gc


a = 1.0


import optweak as ot

ot.add_suggest('name',)
ot.get_suggest('name', default=0.1)

g_trial = None

def tweak(trial)->None:
    """start tweak the trial

    Args:
        trial ([type]): [description]

    Returns:
        [type]: [description]
    """
    global g_trial
    g_trial = trial


class OptunaIntValue():

    def __init__(self, name:str):
        self.name = name

    def __float__(self):
        return 0.5

    def __int__(self):
        return 1

    def __long__(self):
        return 100

    @classmethod
    def list_optuna_parameters(cls):
        pass

