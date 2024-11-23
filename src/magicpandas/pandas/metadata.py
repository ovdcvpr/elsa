from __future__ import annotations

from typing import *
from magicpandas.magic.cached.serialized import Serialized
from pandas.core import generic

if False:
    from magicpandas.pandas.ndframe import NDFrame



class MetaDatas(set[str]):

    def __set_name__(self, owner: NDFrame, name):
        self.__cache__: dict[type[NDFrame], MetaDatas] = {}
        self.__name__ = name

    def __get__(self, instance: object, owner: type[NDFrame]) -> Self:

        if owner not in self.__cache__:
            # todo: add an isinstance for inheriting from bases
            result = {
                key
                for base in owner.__bases__[::-1]
                if (
                        issubclass(base, generic.NDFrame)
                        and hasattr(base, self.__name__)
                )
                # for key, value in getattr(base, self.__name__).items()
                for key in getattr(base, self.__name__)
            }
            result = self.__class__(result)
            result.update({
                key
                for key, value in owner.__dict__.items()
                if isinstance(value, Serialized)
            })
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]
        return result
