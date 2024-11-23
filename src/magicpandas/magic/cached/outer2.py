from __future__ import annotations
from magicpandas.magic.cached.abc import CachedABC

import importlib
import weakref
from functools import *
from typing import *

import magicpandas.util as util



if False:
    import magicpandas as magic
    from magicpandas.magic import Magic


class Cache:
    def __get__(self, instance: Outer, owner: type[Outer]) -> Self:
        self.outer = instance
        self.Outer = owner
        return self

    def __setitem__(self, key, value):
        if (
            value is not None
            and not isinstance(value, weakref.ref)
        ):
            value = weakref.ref(value)
        self.__cache__[key] = value

    def __getitem__(self, key):
        magic = self.outer.__magic__
        cache = magic.__third__.__dict__
        result = cache[key]
        if result is None:
            raise ValueError(f"weakref to {key} in {magic} is None")
        else:
            # todo: unexpected: truth.elsa is not a weakref
            result = result()
        return result

    @property
    def __cache__(self):
        return self.outer.__magic__.__third__.__dict__

    def __delitem__(self, key):
        try:
            del self.__cache__[key]
        except KeyError:
            ...

    def __contains__(self, item):
        return item in self.__cache__

    def __repr__(self):
        return self.outer.__magic__.__third__.__dict__.__repr__()


class Outer(
    CachedABC
):
    cache = Cache()

    @util.weakly.cached_property
    def __magic__(self) -> magic.Magic:
        ...

    def __get__(self: Outer, instance: magic.Magic, owner: type[magic.Magic]) -> Self:
        self.__magic__ = instance
        if instance is None:
            return self
        key = self.__key__
        cache = self.cache
        if key in cache:
            return cache[key]

        outer = instance
        cls = self.cls
        while outer is not None:
            if isinstance(outer, cls):
                break
            outer = outer.__outer__
        else:
            msg = (
                f'Unable to find an outer instance for '
                f'annotation {self.__annotation__} '
            )
            raise ValueError(msg)

        if instance.__order__ == 3:
            self.__set__(instance, outer)
        return outer

    def __set_name__(self, owner: type, name: str):
        self.__name__ = name
        self.__owner__ = owner

    def __init__(self, func):
        self.__annotation__: str = func.__annotations__['return']

    @cached_property
    def cls(self) -> type[Magic]:
        module = importlib.import_module(self.__owner__.__module__)
        cls = util.resolve_annotation(module, self.__annotation__)
        return cls

    @property
    def __trace__(self):
        return (
                self.__magic__.trace
                + self.__name__
        )

    @property
    def __key__(self):
        magic = self.__magic__
        result = (
                magic.__trace__
                - magic.__third__.__trace__
                + self.__name__
        )
        return result

    def __set__(self, instance: magic.Magic, value):
        # self.__outer__ = instance
        self.__magic__ = instance
        if (
            value is not None
            and not isinstance(value, weakref.ref)
        ):
            value = weakref.ref(value)
        if instance.__order__ == 3:
            key = self.__name__
            cache = instance.__dict__
        else:
            key = self.__key__
            cache = self.cache
        cache[key] = value

    def __delete__(self, instance: magic.Magic):
        # self.__outer__ = instance
        self.__magic__ = instance
        del self.cache[self.__key__]
