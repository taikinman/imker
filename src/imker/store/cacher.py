import bz2
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseCacher(ABC):
    @abstractmethod
    def save(self, path: str, obj: Any) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        pass

    def string2path(self, string: str) -> Path:
        return Path(string)

    def path2string(self, path: Path) -> str:
        return path.as_posix()

    @abstractmethod
    def format(self) -> str:
        raise NotImplementedError(
            "format method must be implemented to return file format. If \
                                  you want to use pickle cacher, format method might return 'pkl'."
        )


class PickleCacher(BaseCacher):
    def save(self, path: str, obj: Any) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> Any:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def format(self) -> str:
        return "pkl"


class PickledBz2Cacher(BaseCacher):
    def save(self, path: str, obj: Any) -> None:
        with bz2.BZ2File(path, "w") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> Any:
        data = bz2.BZ2File(path, "rb")
        data = pickle.load(data)
        return data

    def format(self) -> str:
        return "pbz2"


class NumpyCacher(BaseCacher):
    def save(self, path: str, obj: np.ndarray) -> None:
        np.save(path, obj)

    def load(self, path: str) -> Any:
        return np.load(path)

    def format(self) -> str:
        return "npy"


class CSVCacher(BaseCacher):
    def save(slef, path: str, obj: pd.DataFrame) -> None:
        obj.to_csv(path, index=False)

    def load(self, filepath: str) -> Any:
        return pd.read_csv(filepath)

    def format(self) -> str:
        return "csv"
