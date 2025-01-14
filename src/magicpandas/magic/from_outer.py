from __future__ import annotations

from functools import *
from types import *
from typing import *

from magicpandas import util 

if False:
    from .cached import Base


class FromOuter:
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, instance: Base, owner) -> None | Self | MethodDescriptorType:
        if instance is None:
            return self
        result: None | MethodDescriptorType = instance.__dict__.get(self.__name__, None)
        if result is None:
            return result
        result = result.__get__(instance, owner)
        return result

    def __set__(self, instance: Base, value):
        if util.returns(value):
            result = value
        else:
            @wraps(value)
            def result(*args, **kwargs):
                raise NotImplementedError(
                    f"Property-like {value.__qualname__} does not return a value, "
                    f"so it is likely intended to be used only after being set. "
                    f"If the method is to return None, explicitly return None. "
                    f"Otherwise, this is an error from a get before a set."
                )
        instance.__dict__[self.__name__] = result



    def __delete__(self, instance):
        key = self.__name__
        cache = instance.__dict__
        if key in cache:
            del cache[key]
