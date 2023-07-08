import hashlib
import numpy as np
import pandas as pd
import pickle
from inspect import isclass, isfunction, getsource, signature
import re
from collections import OrderedDict
from .container.base import DataContainer


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except:
        return False

def hasfunc(obj, attr):
    return hasattr(obj, attr) and callable(getattr(obj, attr))

def is_dictlike(obj):
    return isinstance(obj, dict) or isinstance(obj, DataContainer) or isinstance(obj, OrderedDict)


def is_func_or_class(obj):
    return isfunction(obj) or isclass(obj)


def get_func_code(func):
    if not callable(func):
        raise ValueError("Input must be a function object")
    return getsource(func)


def get_class_code(cls):
    class_func_names = [k for k, v in cls.__dict__.items() if isfunction(v)]

    super_class_names = ", ".join([b.__name__ for b in cls.__bases__])
    doc = f"class {cls.__name__}({super_class_names}):\n"
    for f in class_func_names:
        doc += f"{get_func_code(getattr(cls, f))}\n"
    return doc


def get_code(obj):
    if isclass(obj):
        return get_class_code(obj)
    elif isfunction(obj):
        return get_func_code(obj)


def hash_data(data):
    if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, dict, tuple, list, set)):
        _data = pickle.dumps(data)
    elif is_func_or_class(data):
        _data = get_code(data).encode()
    else:
        _data = repr(data).encode()
    return hashlib.sha256(_data).hexdigest()


def get_identifier(*args, **kwargs):
    results = ""
    for arg in args:
        try:
            results += hash_data(arg)
        except AttributeError:
            results += " "

    for key, value in kwargs.items():
        results += hash_data(key)
        try:
            results += hash_data(value)
        except AttributeError:
            results += " "
    return hashlib.sha256(results.encode()).hexdigest()


def get_n_returns(returns, cnt=0):
    if isinstance(returns, tuple) or isinstance(returns, list):
        for r in returns:
            if isinstance(r, tuple) or isinstance(r, list):
                cnt = get_n_returns(r, cnt)
            else:
                cnt += 1
    else:
        cnt += 1
    return cnt


def parse_arguments(func):
    params = dict(signature(func).parameters)
    if "self" in params:
        params.pop("self")
    return params


def parse_returns(func):
    comp1 = re.compile("return .+\s*\n*")
    comp2 = re.compile("[\"'][a-zA-Z0-9][\"']")
    code = get_code(func)
    output_keys = list(
        set([r[1:-1] for result in comp1.findall(code) for r in comp2.findall(result)])
    )
    return output_keys
