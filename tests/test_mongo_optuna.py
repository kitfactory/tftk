# from datetime import datetime
from datetime import datetime
import optuna
from optuna._study_direction import StudyDirection
from optuna.trial import FrozenTrial,TrialState
from optuna.storages import BaseStorage
from optuna.storages import InMemoryStorage, RDBStorage
from optuna.distributions import distribution_to_json, json_to_distribution
import sys

from pymongo import MongoClient

class MongoStorage(BaseStorage):
    """MongoDB class for storages.
    This class is not supposed to be directly accessed by library users.
    Storage classes abstract a backend database and provide library internal interfaces to
    read/write history of studies and trials.
    """

    def __init__(self, storage):
        if storage.endswith("/"):
            database = "optuna"
        else:
            storage , database = storage.rsplit("/",1)
        self.client = MongoClient(storage)
        self.db = self.client[database]

    def get_database(self,storage):
        db = MongoClient(storage)

    def create_new_study(self, study_name=None):
        """create new study
        
        Args:
            study_name ([type], optional): [description]. Defaults to None.
        study_id = Column(Integer, primary_key=True)
        study_name = Column(String(MAX_INDEXED_STRING_LENGTH), index=True, unique=True, nullable=False)
        direction = Column(Enum(StudyDirection), nullable=False)
        
        Raises:
            NotImplementedError: [description]
        """
        # type: (Optional[str]) -> int
        # print("create_new_study" , study_name )
        result = self.db.study.find_one({"study_name":study_name})
        if result is None:
            new_id = self.db.study.count_documents({})
            new_study={
                "study_id":new_id,
                "study_name":study_name,
                "direction":structs.StudyDirection.NOT_SET.value,
                "user_attrs":{},
            }
            self.db.study.insert_one(new_study)
            ret = new_id
        else:
            ret =  result["study_id"]
        return ret
        # raise NotImplementedError

    def delete_study(self, study_id):
        # print("delete study " + study_id )
        self.db.study.delete_one({"study_id":study_id})

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None
        # print("set study_user_attr",study_id,key,value)
        study = self.db.find_one({"study_id":study_id})
        study.system_attrs[key] = value
        self.db.study.update_one({"study_id":study_id},{"$set":{"system_attrs":study.sytem_attrs}})

    def set_study_direction(self, study_id, direction):
        # type: (int, structs.StudyDirection) -> None
        # print("set_study_dirction",study_id,direction)
        study =self.db.study.find_one({"study_id":study_id})
        study["study_direction"] = direction.value
        self.db.study.update_one({"study_id":study_id},{'$set': {'direction': direction.value}})

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None
        # print("set_study_system_attr ")
        study = self.db.study.find_one({"study_id":study_id})
        system_attr = study["system_attr"]
        system_attr[key]=value
        self.db.study.find_one_and_update({"study_id":study_id, '$set': {'system_attr': system_attr}})

    # Basic study access

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int
        study = self.db.study.find_one({"study_name":study_name})
        # print("get study id from name",study_name,study)
        if study is None:
            return self.create_new_study(study_name)
        else:
            # print(study["study_id"])
            return study["study_id"]

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int
        # print("get study id from trial",trial_id)
        trial = self.db.trial.find_one({"trial_id": trial_id})
        return trial.study_id

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str
        study = self.db.study.find_one({'study_id': study_id})
        # print("get_study_name_from_id", study_id,study)
        return study["study_name"]

    def get_study_direction(self, study_id):
        # type: (int) -> structs.StudyDirection
        study = self.db.study.find_one({'study_id': study_id})
        return StudyDirection(study["direction"])

    # @abc.abstractmethod
    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]
        # print("get_study_user_attrs")
        study = self.db.study.find_one({"study_id":study_id})
        return study["user_attrs"]

    # @abc.abstractmethod
    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]
        study = self.db.study.find_one({"study_id": study_id})
        return study.system_attrs;

    # @abc.abstractmethod
    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]
        raise NotImplementedError

    # Basic trial manipulation
    """
    trial_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    state = Column(Enum(TrialState), nullable=False)
    value = Column(Float)
    datetime_start = Column(DateTime, default=datetime.now)
    datetime_complete = Column(DateTime)
    """

    """
        Attributes:
        number:
            Unique and consecutive number of :class:`~optuna.trial.Trial` for each
            :class:`~optuna.study.Study`. Note that this field uses zero-based numbering.
        state:
            :class:`TrialState` of the :class:`~optuna.trial.Trial`.
        value:
            Objective value of the :class:`~optuna.trial.Trial`.
        datetime_start:
            Datetime where the :class:`~optuna.trial.Trial` started.
        datetime_complete:
            Datetime where the :class:`~optuna.trial.Trial` finished.
        params:
            Dictionary that contains suggested parameters.
        distributions:
            Dictionary that contains the distributions of :attr:`params`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` set with
            :func:`optuna.trial.Trial.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` internally
            set by Optuna.
        intermediate_values:
            Intermediate objective values set with :func:`optuna.trial.Trial.report`.
        trial_id:
            Optuna's internal identifier of the :class:`~optuna.trial.Trial`. Note that this field
            is not supposed to be used by library users. Instead, please use :attr:`number` and
            :class:`~optuna.study.Study.study_id` to identify a :class:`~optuna.trial.Trial`.
    """

    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[structs.FrozenTrial]) -> int
        # print("create new trial ", study_id, template_trial)
        new_id = self.db.trial.count_documents({})
        new_number = self.db.trial.count_documents({"study_id":study_id})
        new_trial = {
            "trial_id":new_id,
            "number":new_number,
            "study_id":study_id,
            "state":TrialState.RUNNING.value,
            "value":None,
            "datetime_start":datetime.now(),
            "datetime_complete":None,
            "distributions":{},
            "user_attrs":{},
            "system_attrs":{},
            "params":{},
            "intermediate_values":{}
        }
        # print("new_trial id", new_trial)
        self.db.trial.insert_one(new_trial)
        return new_id



    # @abc.abstractmethod
    def set_trial_state(self, trial_id, state):
        self.db.trial.find_one_and_update({"trial_id":trial_id},{"$set":{"state":state.value}})

    # @abc.abstractmethod
    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool
        # print("set trial param trial_id", trial_id, "- param_name",param_name,"param_value_internal",param_value_internal, "distributions",distribution)
        trial = self.db.trial.find_one({"trial_id": trial_id})
        params = trial["params"]
        params[param_name]=param_value_internal
        dist_json = distribution_to_json(distribution)
        dist = trial["distributions"]
        dist[param_name] = dist_json
        self.db.trial.update_one({"trial_id": trial_id},{"$set":{"params":params,"distributions":dist}})

    # @abc.abstractmethod
    def get_trial_number_from_id(self, trial_id):
        trial = self.db.trial.find_one({"trial_id": trial_id})
        return trial["number"]

    # @abc.abstractmethod
    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float
        # print("get_trial_param",trial_id,param_name)
        trial = self.db.trial.find_one({"trial_id": trial_id})
        return trial["params"][param_name]

    # @abc.abstractmethod
    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None
        # print("set_trial_value", trial_id, value)
        # trial = self.db.traial.find_one({"traial_id":trial_id})
        # trial["value"] = value
        self.db.trial.update_one({"trial_id": trial_id},{"$set":{"value":value}})

    # @abc.abstractmethod
    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool
        # print("set_trial_intermediate_value",trial_id,step,intermediate_value)
        trial = self.db.trial.find_one({"trial_id":trial_id})
        inter_val = trial["intermediate_values"]
        inter_val[str(step)] = intermediate_value
        trial_intermediate_value = self.db.trial.find_one_and_update({"traial_id":trial_id},{"$set":{"intermediate_values":inter_val}})
        # raise NotImplementedError

    # @abc.abstractmethod
    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None
        # print("set_trial_user_attr",trial_id,key,value)
        trial = self.db.trial.find_one({"traial_id":trial_id})
        trial["user_attrs"][key]=value
        self.db.trial.update({"traial_id":trial_id},{"$set":{"user_attrs": trial["user_attrs"]}})

    # @abc.abstractmethod
    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None
        # print("set_trial_system_attr *********",trial_id,key,value)

        trial = self.db.trial.find_one({"traial_id":trial_id})
        # print(trial)
        trial["system_attrs"][key]=value
        self.db.trial.update({"traial_id":trial_id},{"$set":{"system_attrs": trial["system_attrs"]}})

    # Basic trial access

    def __to_frozen_trial(self,trial):
        number = trial["number"]
        state = trial["state"]
        datetime_start = trial["datetime_start"]
        datetime_complete = trial["datetime_complete"]
        params = trial["params"]
        distributions = {}
        for p in params:
            distributions[p] = json_to_distribution(trial["distributions"][p])
        user_attrs = trial["user_attrs"]
        system_attrs = trial["system_attrs"]
        trial["intermediate_values"]
        intkey_val = {}
        for k in trial["intermediate_values"]:
            intk = int(k)
            intkey_val[k] = trial["intermediate_values"][k]
        intermediate_values = intkey_val
        trial_id = trial["trial_id"]
        value=trial["value"]
        

        """
def __init__(
        self,
        number,  # type: int
        state,  # type: TrialState
        value,  # type: Optional[float]
        datetime_start,  # type: Optional[datetime.datetime]
        datetime_complete,  # type: Optional[datetime.datetime]
        params,  # type: Dict[str, Any]
        distributions,  # type: Dict[str, BaseDistribution]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        intermediate_values,  # type: Dict[int, float]
        trial_id,  # type: int
        """

        ret = FrozenTrial(
            number=number,
            state= TrialState(state),
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
            intermediate_values=intermediate_values,
            trial_id=trial_id,
            value = value
        )
        return ret

#classoptuna.structs.FrozenTrial(number, state, value, datetime_start, datetime_complete, params, distributions, user_attrs, system_attrs, intermediate_values, trial_id)

    # @abc.abstractmethod
    def get_trial(self, trial_id):
        # print("get_trial")
        # type: (int) -> structs.FrozenTrial
        trial = self.db.trial.find_one({"trial_id": trial_id})
        ret = self.__to_frozen_trial(trial)
        return ret

    # @abc.abstractmethod
    def get_all_trials(self, study_id, deepcopy=False):
        # type: (int) -> List[structs.FrozenTrial]
        # print("get_all_trials")
        trials = self.db.trial.find({"study_id":study_id})
        ret = []
        for trial in trials:
            ft = self.__to_frozen_trial(trial)
            ret.append(ft)
        # print(ret)
        return ret

    # @abc.abstractmethod
    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int
        # print("get_n_trials")
        raise NotImplementedError

    def get_best_trial(self, study_id):
        # print("get_best_trial",study_id)
        # type: (int) -> structs.FrozenTrial
        all_trials = self.get_all_trials(study_id)
        all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError('No trials are completed yet.')

        if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            # print("Maximize")
            max_val = max(all_trials, key=lambda t: t.value)
            # print(max_val)
        # print("Minimize")
        min_val =  min(all_trials, key=lambda t: t.value)
        # print(min_val)

        if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            return max(all_trials, key=lambda t: t.value)
        return min(all_trials, key=lambda t: t.value)

    def get_trial_params(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        return self.get_trial(trial_id).system_attrs

    def remove_session(self):
        # type: () -> None

        pass

    def check_trial_is_updatable(self, trial_id, trial_state):
        # type: (int, structs.TrialState) -> None

        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number))


def get_storage_mongo(storage):
    if storage is None:
        return InMemoryStorage()
    
    if isinstance(storage, str):
        if storage.startswith("mongo"):
            return MongoStorage(storage)
        else:
            return RDBStorage(storage)
    else:
        return storage


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    final_val = (x-2)**2
    for i in range(10):
        internal_val = final_val *  i / 10.0
        trial.report(internal_val,i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return final_val

optuna.storages.get_storage = get_storage_mongo

if __name__ == "__main__":
    storage = "mongodb://localhost/"
    study = optuna.create_study(study_name='distributed-example', storage='mongodb://localhost/',direction='maximize')
    study.optimize(objective, n_trials=20)
