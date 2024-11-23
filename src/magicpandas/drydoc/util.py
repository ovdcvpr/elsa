from __future__ import annotations

# from importlib import import_module

import ast
import os
from functools import *
from pathlib import Path
from types import *
from typing import *

from magicpandas.magic.magic import Magic
from magicpandas.magic.drydoc.drydoc import DryDoc


def module(path: Path | str) -> ModuleType:
    path = (
        path
        .__str__()
        .removeprefix(os.getcwd() + os.sep)
        .replace(os.sep, '.')
        .replace('.py', '')
    )
    result = importlib.import_module(path)
    return result

import importlib.util

def import_module_from_path(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Example usage:
# module = import_module_from_path('/path/to/your/file.py')
# print(module)
