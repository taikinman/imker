import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def str2path(string: str) -> Path:
    return Path(string)
