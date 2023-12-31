from collections import OrderedDict
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import yaml
from pydantic.dataclasses import dataclass
from yaml.dumper import Dumper
from yaml.nodes import MappingNode

from ..container.base import DataContainer
from ..store.cacher import BaseCacher, PickledBz2Cacher


def represent_odict(dumper: Dumper, instance: OrderedDict) -> MappingNode:
    return dumper.represent_mapping("tag:yaml.org,2002:map", instance.items())


yaml.add_representer(OrderedDict, represent_odict)
yaml.add_representer(DataContainer, represent_odict)


@dataclass
class TaskConfig:
    """
    Dataclass contains task, parameters of the task, parameters related to how cache perform,
    and parameter for reproducibility.

    Parameters:
        task : type[Any]
            Task class inheriting BaseTask class or scikit-learn object.
        init_params : dict, optional
            Arguments of task's constructor.
        fit_params : dict, optional
            Parameters required by fit() of the task.
        transform_params : dict, optional
            Parameters required by transform() of the task.
        predict_params : dict, optional
            Parameters required by predict() or predict_proba() of the task.
        repo_dir : Union[str, Path], default='.imker'
            Path to repository storing cache files. This value is overwritten by
            pipeline's constructor arguments if you run the task in pipeline.
        cache_processor : BaseCacher, default=imker.store.cacher.PickledBz2Cacher.
            Serialization or file processor to cache the output of the task.
        cache : bool, default=False.
            If true, the outputs of transform(), predict() or predict_proba() are cached.
        seed : int, default=42.
            Seed for reproducibility.
        verbose : bool, default=True.
            Whether output the processing time of the each tasks.
    """

    task: type[Any]
    init_params: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] = field(default_factory=dict)
    transform_params: dict[str, Any] = field(default_factory=dict)
    predict_params: dict[str, Any] = field(default_factory=dict)
    repo_dir: Path = Path(".imker")
    cache_processor: type[BaseCacher] = PickledBz2Cacher
    cache: bool = False
    seed: int = 42
    verbose: bool = True

    def asdict(self) -> dict[str, Any]:
        result = OrderedDict(asdict(self))
        return result
