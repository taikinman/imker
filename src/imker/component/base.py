from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from ..container.base import DataContainer
from ..inspection import parse_arguments
from ..store.cacher import PickledBz2Cacher
from ..task.task import Task
from ..types import ArrayLike


class _Base(ABC):
    @classmethod
    def load(cls, identifiers: dict[str, str], **kwargs):
        obj = cls(**kwargs)
        for k, v in obj.__dict__.items():
            if isinstance(v, Task):
                v.task = v.load(identifiers[k])
                v.train_status = True
                setattr(obj, k, v)
        return obj

    def set_identifier(self, attr, identifier: Union[Path, str]) -> None:
        getattr(self, attr).identifier = identifier

    def set_repo_dir(self, repo_dir: Union[str, Path]) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                getattr(self, k).repo_dir = repo_dir

    def set_verbose(self, verbose: bool) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                getattr(self, k).verbose = verbose

    def reset_identifier(self) -> None:
        for _k, v in self.__dict__.items():
            if isinstance(v, Task):
                v.reset_identifier()

    @property
    def identifier(self):
        outputs = DataContainer()
        for k, v in self.__dict__.items():
            if isinstance(v, Task):
                outputs[k] = v.identifier
        return outputs


class BasePreProcessor(_Base):
    def forward(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs
    ) -> tuple[ArrayLike, Union[ArrayLike, None]]:
        return X, y

    def __call__(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs
    ) -> tuple[ArrayLike, Union[ArrayLike, None]]:
        return self.forward(X, y, **kwargs)

    def test(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, reset_identifier=True, **kwargs
    ) -> tuple[ArrayLike, Union[ArrayLike, None]]:
        results = self.__call__(X=X, y=y, **kwargs)
        if reset_identifier:
            self.reset_identifier()
        return results


class BasePostProcessor(_Base):
    def forward(self, X: ArrayLike, y: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        return y

    def __call__(self, X: ArrayLike, y: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        return self.forward(X, y)

    def test(
        self, X: ArrayLike, y: dict[str, ArrayLike], reset_identifier=True
    ) -> dict[str, ArrayLike]:
        results = self.__call__(X=X, y=y)
        if reset_identifier:
            self.reset_identifier()
        return results


class BaseModel(_Base):
    @abstractmethod
    def forward(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        proba: bool = False,
        eval_set: Optional[list[tuple[Any, Any]]] = None,
    ) -> dict[str, ArrayLike]:
        raise NotImplementedError

    def __call__(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        proba: bool = False,
        eval_set: Optional[list[tuple[Any, Any]]] = None,
    ) -> dict[str, ArrayLike]:
        kwargs: dict[str, Any] = {}
        args = parse_arguments(self.forward)
        if "proba" in args:
            kwargs["proba"] = proba

        if "eval_set" in args:
            kwargs["eval_set"] = eval_set

        return self.forward(X, y, **kwargs)

    def test(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        proba: bool = False,
        eval_set: Optional[list[tuple[Any, Any]]] = None,
    ) -> dict[str, ArrayLike]:
        results = self.__call__(X=X, y=y, proba=proba, eval_set=eval_set)
        self.reset_identifier()
        return results


class BaseSplitter(_Base):
    @abstractmethod
    def get_n_splits(self) -> int:
        pass

    @abstractmethod
    def split(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Iterator[DataContainer[Any]]:
        pass

    def __call__(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Iterator[DataContainer[Any]]:
        return self.split(X, y)

    def test(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Iterator[DataContainer[Any]]:
        results = self.__call__(X, y)
        self.reset_identifier()
        return results


class BaseScorer(ABC):
    def __init__(self, metrics: list):
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    @abstractmethod
    def calc_metrics(self, y_true: ArrayLike, y_pred: dict[str, ArrayLike]):
        pass

    def __call__(self, y_true: ArrayLike, y_pred: dict[str, ArrayLike]) -> Any:
        return self.calc_metrics(y_true, y_pred)
