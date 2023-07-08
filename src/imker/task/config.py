from dataclasses import dataclass, field, asdict
from typing import Any
from pathlib import Path
import yaml
from collections import OrderedDict

from ..container.base import DataContainer
from ..store.cacher import BaseCacher, PickledBz2Cacher


def represent_odict(dumper, instance):
    return dumper.represent_mapping("tag:yaml.org,2002:map", instance.items())


yaml.add_representer(OrderedDict, represent_odict)
yaml.add_representer(DataContainer, represent_odict)


@dataclass
class TaskConfig:
    task: Any
    init_params: dict = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)
    transform_params: dict = field(default_factory=dict)
    predict_params: dict = field(default_factory=dict)
    repo_dir: Path = Path(".imker")
    cache_processor: BaseCacher = PickledBz2Cacher
    cache: bool = False
    load_from: str = ""
    seed: int = 42

    def asdict(self):
        result = OrderedDict(asdict(self))
        return result
