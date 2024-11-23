from __future__ import annotations

import importlib
import inspect
from typing import Self

if False:
    from magicpandas.magic.magic import Magic


class DelayedImport:
    def __init__(self, func):
        self.func = func

    def __set_name__(self, owner, name):
        self.__name__ = name
        return self

    def __get__(self, instance, owner: type[Magic]):
        attr = self.__name__
        # todo: problem is we're trying to get annotation of child class when
        #   we should get from parent
        annotation = owner.__annotations__[attr]

        if '.' not in annotation:
            name = annotation
            module = inspect.getmodule(owner)
            obj = getattr(module, annotation)
        else:
            module, name = annotation.rsplit('.', 1)
            module = importlib.import_module(module)
            obj = getattr(module, name)
        if (
                obj is Self
                or name == 'Self'
        ):
            obj = owner

        magic = obj(self.func)
        magic.__set_name__(owner, attr)
        setattr(owner, attr, magic)
        result = magic.__get__(instance, owner)
        return result

    def __get__(self, instance, owner: type[Magic]):
        attr = self.__name__
        # todo: problem is we're trying to get annotation of child class when
        #   we should get from parent
        original = owner.__delayed__[attr]
        annotation = original.__annotations__[attr]

        if '.' not in annotation:
            name = annotation
            module = inspect.getmodule(original)
            obj = getattr(module, annotation)
        else:
            module, name = annotation.rsplit('.', 1)
            module = importlib.import_module(module)
            obj = getattr(module, name)
        if (
                obj is Self
                or name == 'Self'
        ):
            obj = original

        magic = obj(self.func)
        magic.__set_name__(original, attr)
        setattr(original, attr, magic)
        result = magic.__get__(instance, owner)
        return result
