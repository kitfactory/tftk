from typing import Callable
from typing import List, Optional

import tensorflow as tf
import optuna
from optuna import Trial
from tftk import Context

class Optuna():

    instance = None

    dict = {}

    @classmethod
    def start_optuna(cls, objective:Callable, storage:str=None, num_of_trials:int=25, direction="minimize"):
        study = optuna.create_study(storage=storage, direction=direction)
        study.optimize(objective,n_trials=num_of_trials)

    @classmethod
    def get_optuna_conext(cls, training_name:str, trial:Trial)->Context:
        """トライアルの開始準備をする。Objective関数の最初に呼び出してください。

        Parameters:
            trial : 開始するトライアルオブジェクト
        
        """
        print("get_context", trial)
        cls.trial = trial
        context = Context.init_context(training_name)
        context[Context.OPTUNA] = True
        context[Context.OPTUNA_TRIAL] = trial
        return context

    @classmethod
    def suggest_choice(cls, name, choice:List):
        instance = Optuna.get_instance()
        u = cls.trial.suggest_choice(choice)
        instance.dict[name] = u

    @classmethod
    def suggest_discrete_uniform(cls, name, low, high, q):
        instance = Optuna.get_instance()
        u = cls.trial.suggest_discrete_uniform(name, low, high, q)
        instance.dict[name] = u

    @classmethod
    def suggest_float(cls, name: str, low: float, high: float, step: Optional[float] = None, log: bool = False):
        instance = Optuna.get_instance()
        u = cls.trial.suggest_float(name, low, high)
        instance.dict[name] = u

    @classmethod
    def get_value(cls, name:str, default=None):
        instance = Optuna.get_instance()
        if name in instance.dict:
            return instance.dict[name]
        else:
            return default

    @classmethod
    def get_instance(cls):
        if cls.instance == None:
            cls.instance = Optuna()
        return cls.instance
