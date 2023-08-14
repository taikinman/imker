import hashlib
import pickle
import re
from collections import OrderedDict
from inspect import getsource, isclass, isfunction, signature
from pickle import PicklingError
from typing import Any, Callable

import numpy as np
import pandas as pd

from .container.base import DataContainer


def is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except PicklingError:
        return False


def hasfunc(obj: Any, attr: str, hasany=False) -> bool:
    if isinstance(attr, str):
        return hasattr(obj, attr) and callable(getattr(obj, attr))
    elif isinstance(attr, (tuple, list)) and hasany:
        return any([hasattr(obj, a) and callable(getattr(obj, a)) for a in attr])
    elif isinstance(attr, (tuple, list)) and not hasany:
        return all([hasattr(obj, a) and callable(getattr(obj, a)) for a in attr])
    else:
        raise AssertionError


def is_dictlike(obj: Any) -> bool:
    return isinstance(obj, dict) or isinstance(obj, DataContainer) or isinstance(obj, OrderedDict)


def is_func_or_class(obj: Any) -> bool:
    return isfunction(obj) or isclass(obj)


def get_func_code(func: Callable) -> str:
    if not callable(func):
        raise ValueError("Input must be a function object")
    return getsource(func)


def get_class_code(cls: type) -> str:
    class_func_names = [k for k, v in cls.__dict__.items() if isfunction(v)]

    super_class_names = ", ".join([b.__name__ for b in cls.__bases__])
    doc = f"class {cls.__name__}({super_class_names}):\n"
    for f in class_func_names:
        doc += f"{get_func_code(getattr(cls, f))}\n"
    return doc


def remove_comment_and_docstring_field_from_code(code: str) -> str:
    pattern_docstring = re.compile("(\"|'){3}((.|\n)*?)(\"|'){3}")
    pattern_hash_in_literal = re.compile("(\"|')+(#)+(\\s)*.+(\"|')+(\n)(!?.)")
    pattern_hash = re.compile("(\"|')*(#)+(\\s)*.+(\"|')*(\n)(!?.)")

    pattern_blankline = re.compile("(\\s)+(\n)+")
    pattern_function = re.compile("((\\s)+(@.+)\n)*(\\s)+def ")
    code = pattern_docstring.sub("", code)

    groups = []
    for match in pattern_hash.finditer(code):
        group = match.group()
        if not pattern_hash_in_literal.match(group):
            st, en = match.span()
            groups.append((st, en - 1))

    for st, en in groups[::-1]:
        code = code[:st] + "\n" + code[en:]

    code = pattern_blankline.sub("\n", code)

    insert_positions = []
    for match in pattern_function.finditer(code):
        insert_positions.append(match.span())

    for st, _ in insert_positions[::-1]:
        code = code[:st] + "\n" + code[st:]

    return code


def get_code(obj: Any, remove_doc_comment: bool = True) -> str:
    if isclass(obj):
        code = get_class_code(obj)
    elif isfunction(obj):
        code = get_func_code(obj)
    else:
        raise

    if remove_doc_comment:
        code = remove_comment_and_docstring_field_from_code(code)

    return code


def hash_data(data: Any) -> str:
    if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, dict, tuple, list, set)):
        _data = pickle.dumps(data)
    elif is_func_or_class(data):
        _data = get_code(data).encode()
    elif isinstance(data, str):
        _data = data.encode()
    else:
        _data = repr(data).encode()
    return hashlib.sha256(_data).hexdigest()


def get_identifier(*args: Any, **kwargs: Any) -> str:
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


def get_n_returns(returns: Any, cnt: int = 0) -> int:
    if isinstance(returns, tuple) or isinstance(returns, list):
        for r in returns:
            if isinstance(r, tuple) or isinstance(r, list):
                cnt = get_n_returns(r, cnt)
            else:
                cnt += 1
    else:
        cnt += 1
    return cnt


def parse_arguments(func: Callable) -> dict[str, Any]:
    params = dict(signature(func).parameters)
    if "self" in params:
        params.pop("self")
    return params


def parse_returns(func: Callable) -> list[set[str]]:
    comp1 = re.compile("return .+\\s*\n*")
    comp2 = re.compile("[\"'][a-zA-Z0-9][\"']")
    code = get_code(func)
    output_keys = list(
        set([r[1:-1] for result in comp1.findall(code) for r in comp2.findall(result)])
    )
    return output_keys
