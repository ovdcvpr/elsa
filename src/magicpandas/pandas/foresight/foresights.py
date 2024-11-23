from __future__ import annotations
import json

from collections import UserDict
from typing import *

import magicpandas.util as util
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.cached.abc import CachedABC
# from magicpandas.pandas.foresight import Foresight
from magicpandas.pandas.foresight.foresight import Foresight


if False:
    from magicpandas.magic.magic import Magic


class Foresights(UserDict[str, Foresight]):
    """
    Names: instances of the different caches that this class inherits
    """

    def __set_name__(self, owner: Magic, name):
        self.__cache__: dict[type[Magic], Self] = {}
        self.__name__ = name

    def __get__(self, instance: Magic, owner: type[Magic]) -> Self:
        INSTANCE = instance
        OWNER = owner
        name = self.__name__
        if owner not in self.__cache__:
            result = self.__cache__[owner] = self.__class__(
                (key, value)
                for base in owner.__bases__
                if issubclass(base, ABCMagic)
                for key, value in getattr(base, name, {}).items()
            )
            result.update({
                key: value
                for key, value in owner.__dict__.items()
                if isinstance(value, Foresight)
            })
        else:
            result = self.__cache__[owner]

        if instance is not None:
            instance = instance.__first__
            cache = instance.__dict__
            if name not in cache:
                cache[name] = result.copy()
            result = cache[name]
        result.__magic__ = INSTANCE
        result.__Magic__ = OWNER

        return result

    @util.weakly.cached_property
    def __magic__(self) -> Magic:
        ...

    @util.weakly.cached_property
    def __Magic__(self) -> Magic:
        ...

    def copy(self):
        return self.__class__(self)

    def __set__(self, instance: Magic, value: Self):
        instance.__dict__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

    def __repr__(self):
        repr_dict = {key: repr(value) for key, value in self.items()}
        return json.dumps(repr_dict, indent=4, separators=(',', ': '))
