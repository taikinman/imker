from .task import Task
from ..inspection import parse_arguments
from ..container.base import DataContainer
from typing import Any, List


class _Base(object):
    def set_identifier(self, attr, identifier: str):
        getattr(self, attr).identifier = identifier

    def set_repo_dir(self, repo_dir: str):
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                getattr(self, k).repo_dir = repo_dir

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


class BaseTask(object):
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any) -> Any:
        raise NotImplementedError


class BaseModel(_Base):
    def forward(self, X: Any, y: Any = None, proba: bool = False):
        raise NotImplementedError

    def __call__(self, X: Any, y: Any = None, proba: bool = False, eval_set: List[tuple] = None):
        kwargs = {}
        args = parse_arguments(self.forward)
        if "proba" in args:
            kwargs["proba"] = proba

        if "eval_set" in args:
            kwargs["eval_set"] = eval_set

        return self.forward(X, y, **kwargs)


class BaseSplitter(_Base):
    def get_n_splits(self):
        raise NotImplementedError

    def split(self, X: Any, y: Any = None, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, X: Any, y: Any = None, *args, **kwargs):
        return self.split(X, y, *args, **kwargs)


class BaseProcessor(_Base):
    def forward(self, X: Any, y: Any = None):
        raise NotImplementedError

    def __call__(self, X: Any, y: Any = None):
        return self.forward(X, y)


class BaseScorer(object):
    def __init__(self, metrics: list):
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def calc_metrics(self, y_true: Any, y_pred: Any):
        raise NotImplementedError

    def __call__(self, y_true: Any, y_pred: Any):
        return self.calc_metrics(y_true, y_pred)
