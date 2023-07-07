import random
import os
import numpy as np
from pathlib import Path


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def str2path(string):
    return Path(string)
