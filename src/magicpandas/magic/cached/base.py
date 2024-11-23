from __future__ import annotations

import inspect
import weakref
from functools import *
from typing import *
from typing import TypeVar

from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.from_outer import FromOuter
from magicpandas.magic.cached.abc import CachedABC

T = TypeVar('T')

if False:
    import magicpandas as magic



def __get__(self, instance: magic.Magic, owner: type[magic.Magic]):
    self.__Outer__ = owner
    if instance is None:
        return self
    key = self.from_outer.__name__
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
        if result is None:
            msg = f'weakref to {self.__name__} in {instance} is None'
            raise ValueError(msg)
        return result

    return result


class Base(
    CachedABC
):
    locals()['__get__'] = __get__
    from_outer = FromOuter()
    __direction__ = 'horizontal'
    __Outer__: type[magic.Magic] = None

    def __init__(self, func):
        self.from_outer = func

    def __set__(self, outer: magic.Magic, value):
        if self.__setter__ is not None:
            value = self.__setter__(value)
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
            if value() is None:
                raise ValueError(
                    f"weakref to {self.__name__} in {outer} is None"
                )
        outer.__cache__[self.__name__] = value

        # if instance is not 3rd order, higher-order copies will inherit this set
        if outer.__order__ != 3:
            outer.__directions__[self.__name__] = 'diagonal'

    def __delete__(self, outer: magic.Magic):
        try:
            del outer.__cache__[self.__name__]
        except KeyError:
            ...
        if self.__deleter__ is not None:
            self.__deleter__()

    def __set_name__(self, owner: magic.Magic, name):
        self.__name__ = name
        self.__Owner__ = owner
        if hasattr(self.from_outer, '__set_name__'):
            self.from_outer.__set_name__(owner, name)
        owner.__directions__[name] = self.__direction__

    def __repr__(self):
        try:
            return (
                f"{self.__class__.__name__} "
                f"{self.__Owner__.__name__}.{self.__name__}"
            )
        except AttributeError:
            return super().__repr__()

    # @builtins.property
    # @property
    # def __wrapped__(self):
    #     return self.from_outer

    # @__wrapped__.setter
    # def __wrapped__(self, value):
    #     if not inspect.isfunction(value):
    #         @wraps(self.from_outer)
    #         def wrapper(*args, **kwargs):
    #             return value
    #
    #         value = wrapper
    #     self.from_outer = value

    @cached_property
    def __setter__(self) -> Callable[[T], T] | None:
        ...

    def setter(self, func):
        self.__setter__ = func
        return self

    @cached_property
    def __deleter__(self) -> Callable[[T], T] | None:
        ...

    def deleter(self, func):
        self.__deleter__ = func
        return self


property = Base
