from ..inspection import (
    parse_arguments,
    get_identifier,
    is_dictlike,
    is_func_or_class,
    hasfunc,
    get_code,
)
from .config import TaskConfig
from ..utils import set_seed
from ..store.cacher import PickledBz2Cacher
from ..container.base import DataContainer

import pandas as pd
import pickle
import copy
from pathlib import Path
import yaml


class Task(object):
    def __init__(self, config: TaskConfig):
        self.cls_name = config.task.__name__

        self.config = config

        self.task = config.task(**self.config.init_params)

        self.__repo_dir = self.config.repo_dir
        self.__cache_processor = self.config.cache_processor
        self.__format = self.__cache_processor.format()
        self.__cache = self.config.cache
        self.__load_from = Path(self.config.load_from)

    def fit(self, X, y=None, *args, **kwargs):
        base_save_dir = self.__repo_dir / "task/fit" / self.cls_name

        set_seed(self.config.seed)
        fit_args = parse_arguments(self.task.fit)

        if "X" in fit_args and "y" in fit_args:
            fit_id_ = self.get_identifier(
                self.task, X, y, *args, **kwargs, **self.config.fit_params
            )
        else:
            fit_id_ = self.get_identifier(self.task, X, *args, **kwargs, **self.config.fit_params)

        save_to = base_save_dir / fit_id_ / f"task.{self.__format}"

        if save_to.exists():
            self.task = PickledBz2Cacher.load(save_to.as_posix())
        else:
            if "X" in fit_args and "y" in fit_args:
                self.task.fit(X, y, *args, **kwargs, **self.config.fit_params)
                fit_id_ = self.get_identifier(
                    self.task, X, y, *args, **kwargs, **self.config.fit_params
                )
            else:
                self.task.fit(X, *args, **kwargs, **self.config.fit_params)
                fit_id_ = self.get_identifier(
                    self.task, X, *args, **kwargs, **self.config.fit_params
                )

            save_to = base_save_dir / fit_id_ / f"task.{PickledBz2Cacher.format()}"

            save_to.parent.mkdir(parents=True, exist_ok=True)
            PickledBz2Cacher.save(save_to.as_posix(), self.task)

            self.dump_config(save_to.parent, "fit")

        self.__load_from = save_to

        return self

    def transform(self, X, y=None):
        base_save_dir = self.__repo_dir / "task/transform" / self.cls_name
        set_seed(self.config.seed)

        args = parse_arguments(self.task.transform)

        if self.__cache:
            if "X" in args and "y" in args:
                transform_id_ = self.get_identifier(self.task, X, y, **self.config.transform_params)
            else:
                transform_id_ = self.get_identifier(self.task, X, **self.config.transform_params)

            save_to = base_save_dir / transform_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                if "X" in args and "y" in args:
                    result = self.task.transform(X, y, **self.config.transform_params)
                    transform_id_ = self.get_identifier(
                        self.task, X, y, **self.config.transform_params
                    )
                else:
                    result = self.task.transform(X, **self.config.transform_params)
                    transform_id_ = self.get_identifier(
                        self.task, X, **self.config.transform_params
                    )

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

    def predict(self, X):
        base_save_dir = self.__repo_dir / "task/predict" / self.cls_name
        set_seed(self.config.seed)

        if self.__cache:
            predict_id_ = self.get_identifier(
                self.task, X, proba=False, **self.config.predict_params
            )

            save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                result = self.task.predict(X, **self.config.predict_params)

                predict_id_ = self.get_identifier(
                    self.task, X, proba=False, **self.config.predict_params
                )

                save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "predict")
        else:
            result = self.task.predict(X)

        return result

    def predict_proba(self, X):
        base_save_dir = self.__repo_dir / "task/predict_proba" / self.cls_name
        set_seed(self.config.seed)

        if self.__cache:
            predict_id_ = self.get_identifier(
                self.task, X, proba=True, **self.config.predict_params
            )

            save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

            if save_to.exists():
                result = self.__cache_processor.load(save_to.as_posix())
            else:
                result = self.task.predict_proba(X, **self.config.predict_params)

                predict_id_ = self.get_identifier(
                    self.task, X, proba=True, **self.config.predict_params
                )

                save_to = base_save_dir / predict_id_ / f"task.{self.__format}"

                if ~save_to.exists():
                    save_to.parent.mkdir(parents=True, exist_ok=True)
                    self.__cache_processor.save(save_to.as_posix(), result)

                self.dump_config(save_to.parent, "predict_proba")
        else:
            result = self.task.predict_proba(X)

        return result

    def get_identifier(self, task, *args, **kwargs):
        argument_hash = Path(get_identifier(*args, **kwargs))
        if self.config.cache_strict:
            src_hash = get_identifier(src=get_code(task))
            state_hash = Path(get_identifier(src=src_hash, state=task.__dict__))
        else:
            state_hash = Path(get_identifier(state=task.__dict__))

        save_to = argument_hash / state_hash
        return save_to

    def get_n_splits(self):
        return self.task.get_n_splits()

    def split(self, X, y=None, *args, **kwargs):
        set_seed(self.config.seed)
        base_save_dir = self.__repo_dir / "task/split" / self.cls_name
        split_id_ = self.get_identifier(self.task, X, y, *args, **kwargs)
        save_to = base_save_dir / split_id_ / f"task.{self.__format}"

        if ~save_to.exists():
            save_to.parent.mkdir(parents=True, exist_ok=True)
            self.__cache_processor.save(save_to.as_posix(), self.task)
            self.dump_config(save_to.parent, "split")
        else:
            self.task = self.__cache_processor.load(save_to)

        self.__load_from = save_to

        for idx_tr, idx_val in self.task.split(X, y, *args, **kwargs):
            outputs = DataContainer()
            outputs.X_train, outputs.y_train = self._split_dataset(X, y, idx_tr)
            outputs.X_valid, outputs.y_valid = self._split_dataset(X, y, idx_val)
            outputs.idx_train = idx_tr
            outputs.idx_valid = idx_val
            yield outputs

    def _split_dataset(self, X, y, idx):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_ = X.iloc[idx]
        elif isinstance(X, list):
            X_ = [X[i] for i in idx]
        else:
            X_ = X[idx]

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_ = y.iloc[idx]
        elif isinstance(y, list):
            y_ = [y[i] for i in idx]
        else:
            y_ = y[idx]

        return X_, y_

    def __call__(self, X, y=None, proba: bool = False, *args, **kwargs):
        if hasfunc(self.task, "predict") or hasfunc(self.task, "predict_proba"):
            if self.__load_from.as_posix() == ".":
                self.fit(X, y, *args, **kwargs)
            else:
                self.task = PickledBz2Cacher.load(self.__load_from.as_posix())

            if proba:
                return self.predict_proba(X)
            else:
                return self.predict(X)

        elif hasfunc(self.task, "transform"):
            if self.__load_from.as_posix() == ".":
                self.fit(X, y, *args, **kwargs)
            else:
                self.task = PickledBz2Cacher.load(self.__load_from.as_posix())

            return self.transform(X, y)

        elif hasfunc(self.task, "split"):
            return self.split(X, y, *args, **kwargs)

        else:
            raise AssertionError

    def test(self, X, y=None, proba: bool = False, *args, **kwargs):
        results = self.__call__(X=X, y=y, proba=proba, *args, **kwargs)
        self.reset_identifier()
        return results

    def reset_identifier(self):
        self.__load_from = Path("")

    def __getstate__(self):
        raise pickle.PicklingError("Task object is not allowed to serialize.")

    def dump_config(self, path: Path, method: str):
        output = DataContainer()
        config = copy.deepcopy(self.config)
        config = self.format_config(config.asdict())

        if method == "fit":
            output.init_params = config.get("init_params")
            output.fit_params = config.get("fit_params")
            output.transform_params = config.get("transform_params")
            output.predict_params = config.get("predict_params")
        elif method == "transform":
            output.transform_params = config.get("transform_params")
            output.load_from = config.get("load_from")
        elif method == "predict" or method == "predict_proba":
            output.predict_params = config.get("predict_params")
            output.load_from = config.get("load_from")
        elif method == "split":
            output.init_params = config.get("init_params")

        output.cache_processor = config.get("cache_processor")
        output.seed = config.get("seed")
        with open(path / "task_config.yml", "w") as f:
            yaml.dump(output, f, indent=4)

    def format_config(self, config: dict):
        for k, v in config.items():
            if is_dictlike(v):
                config[k] = self.format_config(v)
            elif is_func_or_class(v):
                config[k] = v.__qualname__
            elif isinstance(v, list):
                for i in range(len(v)):
                    if is_dictlike(v):
                        config[k] = self.format_config(v)
                    elif is_func_or_class(v[i]):
                        config[k][i] = config[k][i].__qualname__
            elif isinstance(v, Path):
                config[k] = v.as_posix()
        return config

    @property
    def identifier(self):
        return self.__load_from.as_posix()

    @identifier.setter
    def identifier(self, identifier: str):
        self.__load_from = Path(identifier)

    @property
    def repo_dir(self):
        return self.__repo_dir.as_posix()

    @repo_dir.setter
    def repo_dir(self, repo_dir: str):
        self.__repo_dir = Path(repo_dir)
