from typing import Any, List
from abc import ABC, abstractmethod

from ..container.base import DataContainer
from ..inspection import parse_arguments
from ..task.task import Task


class _Base(ABC):
    def set_identifier(self, attr, identifier: str):
        getattr(self, attr).identifier = identifier

    def set_repo_dir(self, repo_dir: str):
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                getattr(self, k).repo_dir = repo_dir

    def set_verbose(self, verbose: bool):
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                getattr(self, k).verbose = verbose

    def dump_params(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                v.dump_params()

    def reset_identifier(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                v.reset_identifier()

    @property
    def identifier(self):
        outputs = DataContainer()
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                outputs[k] = v.identifier
        return outputs


class BaseProcessor(_Base):
    def forward(self, X: Any, y: Any = None, **kwargs):
        return X, y

    def __call__(self, X: Any, y: Any = None, **kwargs):
        return self.forward(X, y, **kwargs)

    def test(self, X: Any, y: Any = None, reset_identifier=True, **kwargs):
        results = self.__call__(X=X, y=y, **kwargs)
        if reset_identifier:
            self.reset_identifier()
        return results


class BaseModel(_Base):
    @abstractmethod
    def forward(self, X: Any, y: Any = None, proba: bool = False, eval_set: List[tuple] = None):
        raise NotImplementedError

    def __call__(self, X: Any, y: Any = None, proba: bool = False, eval_set: List[tuple] = None):
        kwargs = {}
        args = parse_arguments(self.forward)
        if "proba" in args:
            kwargs["proba"] = proba

        if "eval_set" in args:
            kwargs["eval_set"] = eval_set

        return self.forward(X, y, **kwargs)

    def test(self, X: Any, y: Any = None, proba: bool = False, eval_set: List[tuple] = None):
        results = self.__call__(X=X, y=y, proba=proba, eval_set=eval_set)
        self.reset_identifier()
        return results


class BaseSplitter(_Base):
    @abstractmethod
    def get_n_splits(self):
        pass

    @abstractmethod
    def split(self, X: Any, y: Any = None, *args, **kwargs) -> Any:
        pass

    def __call__(self, X: Any, y: Any = None, *args, **kwargs):
        return self.split(X, y, *args, **kwargs)

    def test(self, X: Any, y: Any = None, *args, **kwargs):
        results = self.__call__(X=X, y=y, *args, **kwargs)
        self.reset_identifier()
        return results


class BaseScorer(ABC):
    def __init__(self, metrics: list):
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    @abstractmethod
    def calc_metrics(self, y_true: Any, y_pred: Any):
        pass

    def __call__(self, y_true: Any, y_pred: Any):
        return self.calc_metrics(y_true, y_pred)