import hashlib
import pickle
from typing import Any
from abc import ABC, abstractmethod


class BaseTask(ABC):
    def fit(self, X: Any, y: Any = None):
        return self

    @abstractmethod
    def transform(self, X: Any):
        pass

    def get_identifier(self, X):
        return hashlib.sha256(pickle.dumps(X)).hexdigest()
