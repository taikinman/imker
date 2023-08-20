import pickle
from collections import defaultdict
from copy import deepcopy as dc
from pathlib import Path
from typing import Any, DefaultDict, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ..component.base import (
    BaseModel,
    BasePostProcessor,
    BasePreProcessor,
    BaseScorer,
    BaseSplitter,
)
from ..container.base import DataContainer
from ..task.task import Task
from ..types import ArrayLike


class DefaultScorer(BaseScorer):
    def calc_metrics(self, y_true: ArrayLike, y_pred: dict[str, ArrayLike]) -> pd.Series:
        _results: DefaultDict[str, dict] = defaultdict(dict)
        results = dict()

        for model, pred in y_pred.items():
            for criteria in self.metrics:
                _results[model][criteria.__name__] = criteria(y_true, pred)
            results[model] = pd.Series(_results[model])
        return pd.concat(results)


class Pipeline(object):
    def __init__(
        self,
        repo_dir: Union[str, Path],
        exp_name: Union[str, Path],
        pipeline_name: Union[str, Path],
        verbose: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.exp_name = Path(exp_name)
        self.pipeline_name = Path(pipeline_name)
        self.verbose = verbose

        self.preprocessor: BasePreProcessor = BasePreProcessor()
        self.splitter: BaseSplitter
        self.model: Union[BaseModel, DataContainer[BaseModel]]
        self.oof_preprocessor: DataContainer[BasePreProcessor]
        self.postprocessor: BasePostProcessor
        self.scorer: BaseScorer
        self.__TRAIN_STATUS: bool = False

    def set_preprocessor(self, preprocessor: type[BasePreProcessor]) -> None:
        self.preprocessor = preprocessor()
        self.preprocessor.set_repo_dir(self.repo_dir)
        self.preprocessor.set_verbose(self.verbose)

    def set_splitter(self, splitter: type[BaseSplitter]):
        self.splitter = splitter()
        self.splitter.set_repo_dir(self.repo_dir)
        self.splitter.set_verbose(self.verbose)

    def set_oof_preprocessor(self, oof_preprocessor: type[BasePreProcessor]):
        assert (
            self.splitter is not None
        ), "attribute splitter is not defined. Please set splitter with set_splitter()\
            if you want to set oof preprocessor"
        self.oof_preprocessor = DataContainer(
            {f"fold{i}": oof_preprocessor() for i in range(self.splitter.get_n_splits())}
        )

        for i in range(self.splitter.get_n_splits()):
            self.oof_preprocessor[f"fold{i}"].set_repo_dir(self.repo_dir)
            self.oof_preprocessor[f"fold{i}"].set_verbose(self.verbose)

    def set_model(self, model: type[BaseModel]):
        if self.splitter is None:
            self.model = model()
            self.model.set_repo_dir(self.repo_dir)
            self.model.set_verbose(self.verbose)
        else:
            self.model = DataContainer(
                {f"fold{i}": model() for i in range(self.splitter.get_n_splits())}
            )

            for i in range(self.splitter.get_n_splits()):
                self.model[f"fold{i}"].set_repo_dir(self.repo_dir)
                self.model[f"fold{i}"].set_verbose(self.verbose)

    def set_metrics(self, metrics: list, scorer: Optional[type[BaseScorer]] = None):
        metrics = metrics if isinstance(metrics, list) else [metrics]

        if scorer is None:
            self.scorer = DefaultScorer(metrics)
        else:
            self.scorer = scorer(metrics)

    def set_postprocessor(self, postprocessor: type[BasePostProcessor], *args, **kwargs):
        self.postprocessor = postprocessor(*args, **kwargs)
        self.postprocessor.set_repo_dir(self.repo_dir)
        self.postprocessor.set_verbose(self.verbose)

    def test_preprocessing(self, X, y=None, **kwargs):
        assert hasattr(self, "preprocessor"), "preprocessor is not defined."
        return self.preprocessor.test(X=dc(X), y=dc(y), **dc(kwargs))

    def test_split(self, X, y=None, **kwargs):
        assert hasattr(self, "splitter"), "splitter is not defined."

        for _i, oof in enumerate(
            self.splitter.test(*self.preprocessor.test(X=dc(X), y=dc(y), **dc(kwargs)))
        ):
            yield oof

    def test_oof_preprocessing(self, X, y=None, **kwargs):
        assert hasattr(self, "splitter"), "splitter is not defined."
        assert hasattr(self, "oof_preprocessor"), "oof_preprocessor is not defined."

        for i, oof in enumerate(
            self.splitter.test(*self.preprocessor.test(X=dc(X), y=dc(y), **dc(kwargs)))
        ):
            oof.X_train, oof.y_train = self.oof_preprocessor[f"fold{i}"].test(
                oof.X_train, oof.y_train, reset_identifier=False
            )
            oof.X_valid, oof.y_valid = self.oof_preprocessor[f"fold{i}"].test(
                oof.X_valid, oof.y_valid
            )

            yield oof

    def train(self, X: ArrayLike, y: Optional[ArrayLike], **kwargs):
        self.__TRAIN_STATUS = False

        self.preprocessor.reset_identifier()

        if hasattr(self, "splitter"):
            for i, oof in enumerate(
                self.splitter(*self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs)))
            ):
                if hasattr(self, "oof_preprocessor"):
                    self.oof_preprocessor[f"fold{i}"].reset_identifier()

                    oof.X_train, oof.y_train = self.oof_preprocessor[f"fold{i}"](
                        oof.X_train, oof.y_train
                    )
                    oof.X_valid, oof.y_valid = self.oof_preprocessor[f"fold{i}"](
                        oof.X_valid, oof.y_valid
                    )

                if isinstance(self.model, DataContainer):
                    self.model[f"fold{i}"].reset_identifier()

                    self.model[f"fold{i}"](
                        oof.X_train, oof.y_train, eval_set=[(oof.X_valid, oof.y_valid)]
                    )
                else:
                    raise
        else:
            X_, y_ = self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs))
            self.model.reset_identifier()

            if isinstance(self.model, BaseModel):
                self.model(X_, y_)
            else:
                raise

        self.__TRAIN_STATUS = True
        self.dump()

        return self

    def validate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        proba: bool = False,
        calc_metrics: bool = True,
        **kwargs,
    ):
        assert self.__TRAIN_STATUS, "train must be run before validation"

        preds: DataContainer[DataContainer] = DataContainer()
        __scores: DataContainer[pd.Series] = DataContainer()
        self.__scores: pd.DataFrame

        if hasattr(self, "splitter"):
            for i, oof in enumerate(
                self.splitter(*self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs)))
            ):
                if hasattr(self, "oof_preprocessor"):
                    oof.X_valid, oof.y_valid = self.oof_preprocessor[f"fold{i}"](
                        oof.X_valid, oof.y_valid
                    )

                if isinstance(self.model, DataContainer):
                    val_preds = self.model[f"fold{i}"](oof.X_valid, proba=proba)
                else:
                    raise

                if hasattr(self, "postprocessor") and isinstance(
                    self.postprocessor, BasePostProcessor
                ):
                    val_preds = self.postprocessor(oof.X_valid, val_preds)

                preds[f"fold{i}"] = DataContainer(indices=oof.idx_valid, preds=val_preds)

                if hasattr(self, "scorer") and calc_metrics:
                    __scores[f"fold{i}"] = self.scorer(oof.y_valid, val_preds)

            if calc_metrics:
                self.__scores = pd.concat(__scores, axis=1)

            return self.organize_validation_results(preds)

        else:
            raise AssertionError("splitter is not configured.")

    def inference(self, X_test: ArrayLike, proba=False, **kwargs):
        assert self.__TRAIN_STATUS, "train must be run before inference"

        _preds: DataContainer[dict[str, ArrayLike]] = DataContainer()
        X_, _ = self.preprocessor(X=dc(X_test), y=None, **dc(kwargs))
        if hasattr(self, "splitter"):
            for i in range(self.splitter.get_n_splits()):
                if hasattr(self, "oof_preprocessor"):
                    X_oof, _ = self.oof_preprocessor[f"fold{i}"](X=dc(X_), y=None)
                    if isinstance(self.model, DataContainer):
                        oof_preds = self.model[f"fold{i}"](X_oof, proba=proba)
                    else:
                        raise
                else:
                    if isinstance(self.model, DataContainer):
                        oof_preds = self.model[f"fold{i}"](X_, proba=proba)
                    else:
                        raise
                    oof_preds = DataContainer(oof_preds)

                if hasattr(self, "postprocessor") and isinstance(
                    self.postprocessor, BasePostProcessor
                ):
                    oof_preds = self.postprocessor(X_oof, oof_preds)

                _preds[f"fold{i}"] = oof_preds

            return self.organize_inference_results(_preds)

        else:
            if isinstance(self.model, BaseModel):
                preds = self.model(X_, proba=proba)

            if hasattr(self, "postprocessor") and isinstance(self.postprocessor, BasePostProcessor):
                return self.postprocessor(X=X_, y=preds)

    def get_scores(self):
        return self.__scores

    def organize_validation_results(
        self, val_preds: DataContainer[Any]
    ) -> DataContainer[ArrayLike]:
        indices: np.ndarray = np.array(())
        results: defaultdict[str, Any] = defaultdict(list)
        for _fold, preds in val_preds.items():
            indices = np.hstack([indices, preds.indices])

            for model, out in preds.preds.items():
                results[model].append(out)

        for model in results.keys():
            try:
                results[model] = np.vstack(results[model])
                assert len(results[model]) == len(indices)
            except AssertionError:
                results[model] = np.hstack(results[model])
            results[model] = results[model][np.argsort(indices)]

        return DataContainer(results)

    def organize_inference_results(self, test_preds) -> DataContainer[ArrayLike]:
        results = defaultdict(list)
        for _fold, preds in test_preds.items():
            for model, out in preds.items():
                results[model].append(out)

        for model in results.keys():
            results[model] = np.array(results[model]).mean(axis=0)
        return DataContainer(results)

    @classmethod
    def load(
        cls,
        repo_dir: Union[str, Path],
        exp_name: Union[str, Path],
        pipeline_name: Union[str, Path],
        verbose: bool = True,
        preprocessor: Optional[type[BasePreProcessor]] = None,
        oof_preprocessor: Optional[type[BasePreProcessor]] = None,
        splitter: Optional[type[BaseSplitter]] = None,
        model: Optional[type[BaseModel]] = None,
        postprocessor: Optional[type[BasePostProcessor]] = None,
    ):
        def _check_load_attrs(obj: Any, config: dict[str, Any], tag: str) -> None:
            assert set([k for k, v in obj.__dict__.items() if isinstance(v, Task)]) == set(
                config.keys()
            ), f"Keys didn't match between pipeline config and {tag}"

        repo_dir = Path(repo_dir)
        exp_name = Path(exp_name)
        pipeline_name = Path(pipeline_name)

        config_path = repo_dir / "pipeline" / exp_name / f"{pipeline_name.as_posix()}.yml"

        pipe = cls(
            repo_dir=repo_dir,
            exp_name=exp_name,
            pipeline_name=pipeline_name,
            verbose=verbose,
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "preprocessor" in config:
            assert (
                preprocessor is not None
            ), "pipeline needs preprocessor but you don't pass preprocessor argument"

            pipe.set_preprocessor(preprocessor)

            _check_load_attrs(pipe.preprocessor, config["preprocessor"], "preprocessor")

            for attr, id_ in config["preprocessor"].items():
                pipe.preprocessor.set_identifier(attr, id_)

        if "splitter" in config:
            assert (
                splitter is not None
            ), "pipeline needs splitter but you don't pass splitter argument"

            pipe.set_splitter(splitter)
            _check_load_attrs(pipe.splitter, config["splitter"], "splitter")

            for attr, id_ in config["splitter"].items():
                pipe.splitter.set_identifier(attr, id_)

        if "oof_preprocessor" in config:
            assert (
                oof_preprocessor is not None
            ), "pipeline needs oof_preprocessor but you don't pass oof_preprocessor argument"

            pipe.set_oof_preprocessor(oof_preprocessor)
            for fold, c in config["oof_preprocessor"].items():
                _check_load_attrs(pipe.oof_preprocessor[fold], c, "oof_preprocessor")

                for attr, id_ in c.items():
                    pipe.oof_preprocessor[fold].set_identifier(attr, id_)

        if "model" in config:
            assert model is not None, "pipeline needs model but you don't pass model argument"

            pipe.set_model(model)
            if pipe.splitter is not None:
                for fold, c in config["model"].items():
                    if isinstance(pipe.model, DataContainer):
                        _check_load_attrs(pipe.model[fold], c, "model")

                        for attr, id_ in c.items():
                            pipe.model[fold].set_identifier(attr, id_)
                    else:
                        raise
            else:
                _check_load_attrs(pipe.model, config["model"], "model")
                for attr, id_ in config["model"].items():
                    pipe.model.set_identifier(attr, id_)

        if "postprocessor" in config:
            assert (
                postprocessor is not None
            ), "pipeline needs postprocessor but you don't pass postprocessor argument"

            pipe.set_postprocessor(postprocessor)
            _check_load_attrs(pipe.postprocessor, config["postprocessor"], "postprocessor")
            for attr, id_ in config["postprocessor"].items():
                pipe.postprocessor.set_identifier(attr, id_)

        pipe.train_status = config["train_status"]
        return pipe

    def dump(self) -> None:
        save_dir = self.repo_dir / "pipeline" / self.exp_name
        output: DataContainer[Any] = DataContainer()

        if hasattr(self, "preprocessor"):
            output["preprocessor"] = self.preprocessor.identifier

        if hasattr(self, "splitter"):
            output["splitter"] = self.splitter.identifier

        if hasattr(self, "oof_preprocessor"):
            oof_pp_id: DataContainer[str] = DataContainer()
            for i in range(self.splitter.get_n_splits()):
                oof_pp_id[f"fold{i}"] = self.oof_preprocessor[f"fold{i}"].identifier
            output["oof_preprocessor"] = oof_pp_id

        if hasattr(self, "model"):
            oof_model_id: DataContainer[str] = DataContainer()
            if self.splitter is not None:
                for i in range(self.splitter.get_n_splits()):
                    if isinstance(self.model, DataContainer):
                        oof_model_id[f"fold{i}"] = self.model[f"fold{i}"].identifier
                    else:
                        raise

                output["model"] = oof_model_id
            else:
                if isinstance(self.model, BaseModel):
                    output["model"] = self.model.identifier
                else:
                    raise

        if hasattr(self, "postprocessor"):
            output["postprocessor"] = self.postprocessor.identifier

        output["train_status"] = self.__TRAIN_STATUS

        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"{self.pipeline_name.as_posix()}.yml", "w") as f:
            yaml.dump(output, f, indent=4)

    def __getstate__(self):
        raise pickle.PicklingError("Pipeline object is not allowed to serialize")

    @property
    def train_status(self):
        return self.__TRAIN_STATUS

    @train_status.setter
    def train_status(self, x):
        self.__TRAIN_STATUS = x
