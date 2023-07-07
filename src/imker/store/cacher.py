import bz2
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


class BaseCacher:
    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def string2path(self, string: str):
        return Path(string)

    def path2string(self, path: Path):
        return path.as_posix()

    def format(self):
        raise NotImplementedError(
            "format method must be implemented to return file format. If \
                                  you want to use pickle cacher, format method might return 'pkl'."
        )


class PickleCacher(BaseCacher):
    @staticmethod
    def save(path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def format():
        return "pkl"


class PickledBz2Cacher(BaseCacher):
    @staticmethod
    def save(path, obj):
        with bz2.BZ2File(path, "w") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        data = bz2.BZ2File(path, "rb")
        data = pickle.load(data)
        return data

    @staticmethod
    def format():
        return "pbz2"


class NumpyCacher(BaseCacher):
    @staticmethod
    def save(path, obj):
        np.save(path, obj)

    @staticmethod
    def load(path):
        return np.load(path)

    @staticmethod
    def format():
        return "npy"


class CSVCacher(BaseCacher):
    @staticmethod
    def save(filepath, obj: pd.DataFrame):
        obj.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def format():
        return "csv"
