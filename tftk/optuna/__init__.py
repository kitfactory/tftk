import optuna
from . optuna_util import Optuna
from . mongo import get_storage_mongo

__all__ = ['Optuna']

optuna.storages.get_storage = get_storage_mongo
