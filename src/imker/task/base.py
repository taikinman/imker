import hashlib
import pickle
from typing import Any


class BaseTask(object):
    def fit(self, X: Any, y: Any = None):
        return self

    def get_identifier(self, X):
        return hashlib.sha256(pickle.dumps(X)).hexdigest()
