from __future__ import annotations

import weakref
from functools import cached_property
from types import FunctionType
from typing import *
from typing import TypeVar

from magicpandas.magic import cached as _cached
from magicpandas.magic import magic
from magicpandas.magic.abc import ABCMagic

if False:
    from .ndframe import NDFrame

T = TypeVar('T')

def __get__(self, instance: NDFrame, owner):
    if instance is None:
        return self
    key = self.__name__
    cache = instance.__cache__
    if key not in cache:
        value = (
            self
            .from_outer
            .__get__(instance, owner)
            .__call__()
        )
        self.__set__(instance, value)
    result = cache[key]
    if isinstance(result, weakref.ref):
        result = result()
    return result


class Base(_cached.cached.Base):
    """Base does not use outer, owner, etc."""
    locals()['__get__'] = __get__

    def __init__(self, __func__: FunctionType | magic.Magic, *args, **kwargs):
        if isinstance(__func__, _cached.cached.Base):
            self.__dict__.update(__func__.__dict__)
        else:
            super().__init__(__func__, *args, **kwargs)

    def __set__(self, instance: NDFrame, value):
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
        if self.__setter__ is not None:
            value = self.__setter__(value)
        instance.__cache__[self.__name__] = value

        # if instance is not 3rd order, higher-order copies will inherit this set
        if instance.__order__ != 3:
            instance.__directions__[self.__name__] = 'diagonal'

    def __delete__(self, instance: NDFrame):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...

    @cached_property
    def __setter__(self) -> Callable[[T], T] | None:
        ...

    def setter(self, func):
        self.__setter__ = func
        return self


# class cached:
#     # property = Base
#     property = Magic
#
#     class base:
#         property = Base
#
#     class magic:
#         property = Magic
#
#     class root:
#         property = Root
#
#     class cmdline:
#         property = Magic
#
#     class volatile:
#         property = Volatile
#
#     class diagonal:
#         property = Diagonal
#
#     class frame:
#         property = Frame
#
#     class serialized:
#         property = Serialized
#
#     class public:
#         property = Public
#
#     class outer:
#         property = Outer
#
#     class local:
#         property = Local
#
