import pickle
from collections import defaultdict
from copy import deepcopy as dc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..component.base import BaseModel, BaseProcessor, BaseScorer, BaseSplitter
from ..container.base import DataContainer
from ..task.task import Task


class DefaultScorer(BaseScorer):
    def calc_metrics(self, y_true, y_pred):
        _results = defaultdict(dict)
        results = dict()

        for model, pred in y_pred.items():
            for criteria in self.metrics:
                _results[model][criteria.__name__] = criteria(y_true, pred)
            results[model] = pd.Series(_results[model])
        return pd.concat(results)


class Pipeline(object):
    def __init__(self, repo_dir: str, exp_name: str, pipeline_name: str, verbose: bool = True):
        self.repo_dir = Path(repo_dir)
        self.exp_name = Path(exp_name)
        self.pipeline_name = Path(pipeline_name)
        self.verbose = verbose

        self.preprocessor = None
        self.splitter = None
        self.oof_preprocessor = None
        self.model = None
        self.postprocessor = None
        self.scorer = None
        self.__TRAIN_STATUS = False

    def set_preprocessor(self, preprocessor: BaseProcessor, *args, **kwargs):
        self.preprocessor = preprocessor(*args, **kwargs)
        self.preprocessor.set_repo_dir(self.repo_dir)
        self.preprocessor.set_verbose(self.verbose)

    def set_splitter(self, splitter: BaseSplitter, *args, **kwargs):
        self.splitter = splitter(*args, **kwargs)
        self.splitter.set_repo_dir(self.repo_dir)
        self.splitter.set_verbose(self.verbose)

    def set_oof_preprocessor(self, oof_preprocessor: BaseProcessor, *args, **kwargs):
        assert (
            self.splitter is not None
        ), "attribute splitter is not defined. Please set splitter with set_splitter()\
            if you want to set oof preprocessor"
        self.oof_preprocessor = DataContainer(
            {
                f"fold{i}": oof_preprocessor(*args, **kwargs)
                for i in range(self.splitter.get_n_splits())
            }
        )

        for i in range(self.splitter.get_n_splits()):
            self.oof_preprocessor[f"fold{i}"].set_repo_dir(self.repo_dir)
            self.oof_preprocessor[f"fold{i}"].set_verbose(self.verbose)

    def set_model(self, model: BaseModel, *args, **kwargs):
        if self.splitter is None:
            self.model = model(*args, **kwargs)
            self.model.set_repo_dir(self.repo_dir)
            self.model.set_verbose(self.verbose)
        else:
            self.model = DataContainer(
                {f"fold{i}": model(*args, **kwargs) for i in range(self.splitter.get_n_splits())}
            )

            for i in range(self.splitter.get_n_splits()):
                self.model[f"fold{i}"].set_repo_dir(self.repo_dir)
                self.model[f"fold{i}"].set_verbose(self.verbose)

    def set_metrics(self, metrics: list, scorer: BaseScorer = None):
        metrics = metrics if isinstance(metrics, list) else [metrics]

        if scorer is None:
            self.scorer = DefaultScorer(metrics)
        else:
            self.scorer = scorer(metrics)

    def set_postprocessor(self, postprocessor: BaseProcessor, *args, **kwargs):
        self.postprocessor = postprocessor(*args, **kwargs)
        self.postprocessor.set_repo_dir(self.repo_dir)
        self.postprocessor.set_verbose(self.verbose)

    def test_preprocessing(self, X, y=None, **kwargs):
        return self.preprocessor.test(X=dc(X), y=dc(y), **dc(kwargs))

    def test_split(self, X, y=None, **kwargs):
        if self.splitter is not None:
            for i, oof in enumerate(
                self.splitter.test(*self.preprocessor.test(X=dc(X), y=dc(y), **dc(kwargs)))
            ):
                yield oof
        else:
            raise AssertionError("splitter is not defined")

    def test_oof_preprocessing(self, X, y=None, **kwargs):
        if self.splitter is not None:
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
        else:
            raise AssertionError("splitter is not defined")

    def train(self, X, y=None, **kwargs):
        self.__TRAIN_STATUS = False

        if self.preprocessor is None:
            self.preprocessor = BaseProcessor()

        self.preprocessor.reset_identifier()

        if self.splitter is not None:
            for i, oof in enumerate(
                self.splitter(*self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs)))
            ):
                if self.oof_preprocessor is not None:
                    self.oof_preprocessor[f"fold{i}"].reset_identifier()

                    oof.X_train, oof.y_train = self.oof_preprocessor[f"fold{i}"](
                        oof.X_train, oof.y_train
                    )
                    oof.X_valid, oof.y_valid = self.oof_preprocessor[f"fold{i}"](
                        oof.X_valid, oof.y_valid
                    )

                self.model[f"fold{i}"].reset_identifier()

                self.model[f"fold{i}"](
                    oof.X_train, oof.y_train, eval_set=[(oof.X_valid, oof.y_valid)]
                )
        else:
            X_, y_ = self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs))
            self.model.reset_identifier()
            self.model(X_, y_)

        self.__TRAIN_STATUS = True
        self.dump()

        return self

    def validate(self, X, y=None, proba=False, calc_metrics: bool = True, **kwargs):
        assert self.__TRAIN_STATUS, "train must be run before validation"

        preds = DataContainer()
        self.__scores = DataContainer()

        if self.splitter is not None:
            for i, oof in enumerate(
                self.splitter(*self.preprocessor(X=dc(X), y=dc(y), **dc(kwargs)))
            ):
                if self.oof_preprocessor is not None:
                    oof.X_valid, oof.y_valid = self.oof_preprocessor[f"fold{i}"](
                        oof.X_valid, oof.y_valid
                    )

                val_preds = self.model[f"fold{i}"](oof.X_valid, proba=proba)
                val_preds = DataContainer(val_preds)

                if self.postprocessor is not None:
                    val_preds = self.postprocessor(oof.X_valid, val_preds)

                preds[f"fold{i}"] = DataContainer(indices=oof.idx_valid, preds=val_preds)

                if self.scorer is not None and calc_metrics:
                    self.__scores[f"fold{i}"] = self.scorer(oof.y_valid, val_preds)

            if calc_metrics:
                self.__scores = pd.concat(self.__scores, axis=1)

            preds = self.organize_validation_results(preds)

            return preds

        else:
            raise AssertionError("splitter is not configured.")

    def inference(self, X_test, proba=False, **kwargs):
        assert self.__TRAIN_STATUS, "train must be run before inference"

        preds = DataContainer()
        X_, _ = self.preprocessor(X=dc(X_test), y=None, **dc(kwargs))
        if self.splitter is not None:
            for i in range(self.splitter.get_n_splits()):
                if self.oof_preprocessor is not None:
                    X_oof, _ = self.oof_preprocessor[f"fold{i}"](X=dc(X_), y=None)
                    oof_preds = self.model[f"fold{i}"](X_oof, proba=proba)
                else:
                    oof_preds = self.model[f"fold{i}"](X_, proba=proba)
                    oof_preds = DataContainer(oof_preds)

                if self.postprocessor is not None:
                    oof_preds = self.postprocessor(X_oof, oof_preds)

                preds[f"fold{i}"] = oof_preds

            preds = self.organize_inference_results(preds)

        else:
            preds = self.model(X_, proba=proba)
            preds = self.postprocessor(X=X_, y=preds)

        return preds

    def get_scores(self):
        return self.__scores

    def organize_validation_results(self, val_preds):
        indices = np.array(())
        results = defaultdict(list)
        for fold, preds in val_preds.items():
            indices = np.hstack([indices, preds.indices])

            for model, out in preds.preds.items():
                results[model].append(out)

        for model in results.keys():
            try:
                results[model] = np.vstack(results[model])
                assert len(results[model]) == len(indices)
            except:
                results[model] = np.hstack(results[model])
            results[model] = results[model][np.argsort(indices)]

        return DataContainer(results)

    def organize_inference_results(self, test_preds):
        results = defaultdict(list)
        for fold, preds in test_preds.items():
            for model, out in preds.items():
                results[model].append(out)

        for model in results.keys():
            results[model] = np.array(results[model]).mean(axis=0)
        return DataContainer(results)

    @classmethod
    def load(
        cls,
        repo_dir: str,
        exp_name: str,
        pipeline_name: str,
        verbose: bool = True,
        preprocessor: BaseProcessor = None,
        oof_preprocessor: BaseProcessor = None,
        splitter: BaseSplitter = None,
        model: BaseModel = None,
        postprocessor: BaseProcessor = None,
    ):
        def _check_load_attrs(obj, config, tag):
            assert set([k for k, v in obj.__dict__.items() if isinstance(v, Task)]) == set(
                config.keys()
            ), f"Keys didn't match between pipeline config and {tag}"

        repo_dir = Path(repo_dir)
        exp_name = Path(exp_name)
        pipeline_name = Path(pipeline_name)

        config_path = repo_dir / "pipeline" / exp_name / f"{pipeline_name.as_posix()}.yml"

        pipe = cls(
            repo_dir=repo_dir, exp_name=exp_name, pipeline_name=pipeline_name, verbose=verbose
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
                    _check_load_attrs(pipe.model[fold], c, "model")

                    for attr, id_ in c.items():
                        pipe.model[fold].set_identifier(attr, id_)
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

    def dump(self):
        save_dir = self.repo_dir / "pipeline" / self.exp_name
        output = DataContainer()

        if self.preprocessor is not None:
            output["preprocessor"] = self.preprocessor.identifier

        if self.splitter is not None:
            output["splitter"] = self.splitter.identifier

        if self.oof_preprocessor is not None:
            oof_pp_id = DataContainer()
            for i in range(self.splitter.get_n_splits()):
                oof_pp_id[f"fold{i}"] = self.oof_preprocessor[f"fold{i}"].identifier
            output["oof_preprocessor"] = oof_pp_id

        if self.model is not None:
            oof_model_id = DataContainer()
            if self.splitter is not None:
                for i in range(self.splitter.get_n_splits()):
                    oof_model_id[f"fold{i}"] = self.model[f"fold{i}"].identifier

                output["model"] = oof_model_id
            else:
                output["model"] = self.model.identifier

        if self.postprocessor is not None:
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
