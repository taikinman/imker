import hashlib
import pickle
from typing import Optional

from ..types import ArrayLike


class BaseTask(object):
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        return self

    def get_identifier(self, X: ArrayLike) -> str:
        return hashlib.sha256(pickle.dumps(X)).hexdigest()
