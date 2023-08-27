import copy
import hashlib
import pickle
from typing import Optional

from ..inspection import get_code, hasfunc
from ..types import ArrayLike


class BaseTask(object):
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        return self

    def get_identifier(self, X: ArrayLike) -> str:
        return hashlib.sha256(pickle.dumps(X)).hexdigest()

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if hasfunc(cls, ("transform", "split", "predict", "predict_proba", "forward"), hasany=True):
            return self
        else:
            raise NotImplementedError(
                "Task hasn't any required method, you should implement one of the transform(), \
split(), predict() or predict_proba(), and forward()"
            )

    def get_code(self):
        return get_code(self.__class__)
