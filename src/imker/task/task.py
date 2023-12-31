import copy
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ..container.base import DataContainer
from ..inspection import (
    get_code,
    get_identifier,
    hasfunc,
    is_builtin_class_instance,
    is_dictlike,
    is_func_or_class,
    parse_arguments,
)
from ..store.cacher import PickledBz2Cacher
from ..types import ArrayLike
from ..utils import set_seed
from .config import TaskConfig


class Task(object):
    def __init__(self, config: TaskConfig) -> None:
        assert isinstance(
            config, TaskConfig
        ), "config argument must be the instance of the TaskConfig"

        self.cls_name = config.task.__name__
        self.config = config

        self.__repo_dir = self.config.repo_dir
        self.__cache_processor = self.config.cache_processor()
        self.__format = self.__cache_processor.format()
        self.__cache = self.config.cache
        self.__identifier = Path("")
        self.__verbose = self.config.verbose
        self.__train_status: bool = False

        self.task = config.task(**self.config.init_params)

    def timer(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args: tuple[Any], **kargs: dict[str, Any]) -> Any:
            start = time.time()
            result = func(self, *args, **kargs)
            elapsed_time = str(round(time.time() - start, 4))

            if self.__verbose:
                task_name = self.cls_name.ljust(30, " ")
                func_name = func.__name__.rjust(15, " ")
                elapsed_time = elapsed_time.ljust(len(elapsed_time.split(".")[0]) + 5, "0")
                print(f"{task_name} : {func_name} process takes {elapsed_time} [sec]")
            return result

        return wrapper

    @timer
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, *args, **kwargs):
        base_save_dir = self.__repo_dir / "task/fit" / self.cls_name

        set_seed(self.config.seed)
        fit_args = parse_arguments(self.task.fit)

        if "X" in fit_args and "y" in fit_args:
            fit_id_ = self.get_identifier(X, y, *args, **kwargs, **self.config.fit_params)
        else:
            fit_id_ = self.get_identifier(X, *args, **kwargs, **self.config.fit_params)

        save_to = base_save_dir / fit_id_ / f"task.{self.__format}"

        if save_to.exists():
            print(f"{self.cls_name} : load task...")
            self.task = PickledBz2Cacher().load(save_to.as_posix())
        else:
            if "X" in fit_args and "y" in fit_args:
                self.task.fit(X, y, *args, **kwargs, **self.config.fit_params)
            else:
                self.task.fit(X, *args, **kwargs, **self.config.fit_params)

            save_to.parent.mkdir(parents=True, exist_ok=True)
            PickledBz2Cacher().save(save_to.as_posix(), self.task)

            self.dump_config(save_to.parent, "fit")

        self.__identifier = save_to
        self.__train_status = True
        return self

    @timer
    def transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        result: ArrayLike

        base_save_dir = self.__repo_dir / "task/transform" / self.cls_name
        set_seed(self.config.seed)

        args = parse_arguments(self.task.transform)

        if self.__cache:
            if "X" in args and "y" in args:
                transform_id_ = self.get_identifier(X, y, **self.config.transform_params)
            else:
                transform_id_ = self.get_identifier(X, **self.config.transform_params)

            save_to = base_save_dir / transform_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                if "X" in args and "y" in args:
                    result = self.task.transform(X, y, **self.config.transform_params)
                else:
                    result = self.task.transform(X, **self.config.transform_params)

                save_to = base_save_dir / transform_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "transform")
        else:
            if "X" in args and "y" in args:
                result = self.task.transform(X, y, **self.config.transform_params)
            else:
                result = self.task.transform(X, **self.config.transform_params)

        return result

    @timer
    def predict(self, X: ArrayLike) -> ArrayLike:
        result: ArrayLike
        base_save_dir = self.__repo_dir / "task/predict" / self.cls_name
        set_seed(self.config.seed)

        if self.__cache:
            predict_id_ = self.get_identifier(X, proba=False, **self.config.predict_params)

            save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                result = self.task.predict(X, **self.config.predict_params)

                save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "predict")
        else:
            result = self.task.predict(X, **self.config.predict_params)

        return result

    @timer
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        result: ArrayLike
        base_save_dir = self.__repo_dir / "task/predict_proba" / self.cls_name
        set_seed(self.config.seed)

        if self.__cache:
            predict_id_ = self.get_identifier(X, proba=True, **self.config.predict_params)

            save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                result = self.task.predict_proba(X, **self.config.predict_params)

                save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "predict_proba")
        else:
            result = self.task.predict_proba(X, **self.config.predict_params)

        return result

    @timer
    def forward(self, X: ArrayLike) -> ArrayLike:
        result: ArrayLike
        base_save_dir = self.__repo_dir / "task/forward" / self.cls_name
        set_seed(self.config.seed)

        if self.__cache:
            forward_id_ = self.get_identifier(X, proba=False, **self.config.predict_params)

            save_to = base_save_dir / forward_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                result = self.task.forward(X, **self.config.predict_params)

                save_to = base_save_dir / forward_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "forward")
        else:
            result = self.task.forward(X, **self.config.predict_params)

        return result

    def get_identifier(self, *args: Any, **kwargs: Any) -> Path:
        argument_hash: Path = Path(get_identifier(*args, **kwargs))
        state = copy.deepcopy(self.task.__dict__)

        remove = []
        for key in state.keys():
            if key.endswith("_"):
                remove.append(key)

        for key in remove:
            del state[key]

        if hasfunc(self.task, "get_code"):
            state_hash: Path = Path(get_identifier(src=self.task.get_code(), state=state))
        else:
            state_hash: Path = Path(  # type: ignore[no-redef]
                get_identifier(src=get_code(self.config.task), state=state)
            )
        save_to = argument_hash / state_hash
        return save_to

    def get_n_splits(self):
        return self.task.get_n_splits()

    @timer
    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        stratify: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Iterator[DataContainer[Any]]:
        set_seed(self.config.seed)
        base_save_dir = self.__repo_dir / "task/split" / self.cls_name
        split_id_ = self.get_identifier(X, y, stratify, groups)
        save_to = base_save_dir / split_id_ / f"task.{self.__format}"

        if ~save_to.exists():
            save_to.parent.mkdir(parents=True, exist_ok=True)
            self.__cache_processor.save(save_to.as_posix(), self.task)
            self.dump_config(save_to.parent, "split")
        else:
            self.task = self.__cache_processor.load(save_to.as_posix())

        self.__identifier = save_to

        if stratify is None:
            folds = self.task.split(X, y, groups=groups)
        else:
            folds = self.task.split(X, stratify, groups=groups)

        for idx_tr, idx_val in folds:
            oof: DataContainer[Any] = DataContainer()
            oof.X_train, oof.y_train = self._split_dataset(X, y, idx_tr)
            oof.X_valid, oof.y_valid = self._split_dataset(X, y, idx_val)
            oof.idx_train = idx_tr
            oof.idx_valid = idx_val
            yield oof

    def _split_dataset(
        self, X: ArrayLike, y: Optional[ArrayLike], idx: list[int]
    ) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_: Union[pd.DataFrame, pd.Series] = X.loc[idx]
        elif isinstance(X, list):
            X_: list[Any] = [X[i] for i in idx]  # type: ignore
        elif isinstance(X, np.ndarray):
            X_ = X[idx]  # type: ignore
        else:
            raise

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_ = y.loc[idx]
        elif isinstance(y, list):
            y_: list[Any] = [y[i] for i in idx]  # type: ignore
        elif isinstance(y, np.ndarray):
            y_ = y[idx]  # type: ignore
        else:
            raise

        return X_, y_

    def __call__(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        proba: bool = False,
        run_predict_on_fit_end: bool = False,
        *args,
        **kwargs,
    ):
        if hasfunc(self.task, "predict") or hasfunc(self.task, "predict_proba"):
            if not self.__train_status:
                self.fit(X, y, *args, **kwargs)

                if run_predict_on_fit_end:
                    if proba:
                        return self.predict_proba(X)
                    else:
                        return self.predict(X)

            else:
                if proba:
                    return self.predict_proba(X)
                else:
                    return self.predict(X)

        elif hasfunc(self.task, "transform"):
            if not self.__train_status:
                self.fit(X, y, *args, **kwargs)

            return self.transform(X, y)

        elif hasfunc(self.task, "forward"):
            if not self.__train_status:
                self.fit(X, y, *args, **kwargs)

                if run_predict_on_fit_end:
                    return self.forward(X)
            else:
                return self.forward(X)

        elif hasfunc(self.task, "split"):
            return self.split(X, y, *args, **kwargs)

        else:
            raise AssertionError

    def test(
        self,
        X: ArrayLike,
        y: ArrayLike,
        proba: bool = False,
        *args,
        **kwargs,
    ) -> ArrayLike:
        results: ArrayLike
        results = self.__call__(X, y, proba, *args, **kwargs)
        self.reset_identifier()
        return results

    def reset_identifier(self) -> None:
        self.__train_status = False
        self.__identifier = Path("")
        self.task = self.config.task(**self.config.init_params)

    def __getstate__(self):
        raise pickle.PicklingError("Task object is not allowed to serialize.")

    def dump_config(self, path: Path, method: str) -> None:
        output: DataContainer[Any] = DataContainer()
        _output = self.format_config(self.config.asdict())

        if method == "fit":
            output.init_params = _output.get("init_params")
            output.fit_params = _output.get("fit_params")
            output.transform_params = _output.get("transform_params")
            output.predict_params = _output.get("predict_params")
        elif method == "transform":
            output.transform_params = _output.get("transform_params")
            output.load_from = _output.get("load_from")
        elif method == "predict" or method == "predict_proba" or method == "forward":
            output.predict_params = _output.get("predict_params")
            output.load_from = _output.get("load_from")
        elif method == "split":
            output.init_params = _output.get("init_params")

        output.cache_processor = _output.get("cache_processor")
        output.seed = _output.get("seed")

        with open(path / "task_config.yml", "w") as f:
            yaml.dump(output, f, indent=4)

    def format_config(self, config: Union[dict[str, Any], DataContainer]) -> dict[str, Any]:
        def format_list(data: list) -> list:
            for i in range(len(data)):
                if is_dictlike(data[i]):
                    data[i] = self.format_config(data[i])
                elif is_func_or_class(data[i]):
                    try:
                        data[i] = data[i].__qualname__
                    except AttributeError:
                        data[i] = type(data[i]).__qualname__
                elif isinstance(data[i], Path):
                    data[i] = data[i].as_posix()
                elif isinstance(data[i], list):
                    data[i] = format_list(data[i])
                elif not is_builtin_class_instance(v):
                    data[i] = type(data[i]).__module__

            return data

        for k, v in config.items():
            if is_dictlike(v):
                config[k] = self.format_config(v)
            elif is_func_or_class(v):
                try:
                    config[k] = v.__qualname__
                except AttributeError:
                    config[k] = type(v).__qualname__
            elif isinstance(v, Path):
                config[k] = v.as_posix()
            elif isinstance(v, list):
                config[k] = format_list(v)
            elif not is_builtin_class_instance(v):
                config[k] = type(v).__module__

        return config

    @property
    def identifier(self) -> str:
        return self.__identifier.as_posix()

    @identifier.setter
    def identifier(self, identifier: Union[Path, str]) -> None:
        self.__identifier = Path(identifier)

    @property
    def train_status(self):
        return self.__train_status

    @train_status.setter
    def train_status(self, flg: bool):
        self.__train_status = flg

    @property
    def repo_dir(self) -> str:
        return self.__repo_dir.as_posix()

    @repo_dir.setter
    def repo_dir(self, repo_dir: Union[Path, str]) -> None:
        self.__repo_dir = Path(repo_dir)

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, verbose: bool) -> None:
        self.__verbose = verbose

    @staticmethod
    def load(path: str):
        return PickledBz2Cacher().load(path)
