from __future__ import annotations

from functools import *
from typing import *
import weakref


class WeakProperty:
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, func: Callable):
        update_wrapper(self, func)
        self.__func__ = func

    def __get__(self, instance: object, owner: type):
        key = self.__name__
        cache = instance.__dict__
        if key in cache:
            return cache[key]()
        result = self.__func__(instance)
        if (
            result is not None
            and not isinstance(result, weakref.ReferenceType)
        ):
            result = weakref.ref(result)
        cache[key] = result
        return result

    def __set__(self, instance, value):
        key = self.__name__
        if (
            value is not None
            and not isinstance(value, weakref.ReferenceType)
        ):
            value = weakref.ref(value)
        instance.__dict__[key] = value

    def __delete__(self, instance):
        del instance.__dict__[self.__name__]


class weakly:
    cached_property = WeakProperty
