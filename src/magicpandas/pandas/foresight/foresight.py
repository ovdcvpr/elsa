from __future__ import annotations

import functools
import inspect
from functools import *
from types import *
from typing import *

import numpy as np
import pandas as pd

import magicpandas.util as util
from collections import UserDict
from copy import copy
from magicpandas.pandas.foresight.util import write
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.order import Order

if False:
    from magicpandas.magic.magic import Magic
    from magicpandas.pandas.ndframe import NDFrame


class MetaForesight(type):
    def __getitem__(self, item) -> type[Foresight]:
        return functools.partial(self, item=item)


class Foresight(
    # ABCMagic
):
    # __order__ = Order.third
    def __bool__(self):
        return bool(self.__doc__)

    def __init__(
            self,
            func: Callable[[NDFrame], Any],
            item,
            attr: NDFrame,
    ):
        update_wrapper(self, func)
        self.__func__ = func
        self.__item__ = item
        self.__attr__ = attr

    def __get__(self, instance: NDFrame, owner: type[NDFrame]):
        self.__magic__ = instance
        self.__Magic__ = owner
        return self

    def __set_name__(self, cls: type[NDFrame], name):
        self.__name__ = name
        self.__cls__ = cls

        if self.__attr__ is not None:
            self.__attr__.__foresights__[name] = self
        else:
            ...



    # todo: we need to get the string
    def __call__(self):

        magic = self.__magic__
        if magic is None:
            raise ValueError('magic is None')
        item = self.__item__
        file = inspect.getfile(self.__class__)
        cls = self.__class__.__name__
        func = self.__name__
        attr = self.__attr__

        if attr is not None:
            magic = getattr(magic, attr.__name__)
        if item is not None:
            magic = magic[item]
        docstring = magic.__repr__()
        write(file=file, cls=cls, func=func, docstring=docstring)


class ForesightFactory:
    __item__: Any = None
    __attr__: Magic = None

    def __getitem__(self, item) -> Self:
        result = copy(self)
        result.__item__ = item
        return result

    def __call__(self, func) -> Foresight:
        result = Foresight(
            func=func,
            item=self.__item__,
            attr=self.__attr__,
        )
        return result

    def __get__(self, instance, owner):
        if instance is None:
            return self
        result = copy(self)
        result.__attr__ = instance
        return result


foresight = ForesightFactory()

locals()['foresight'] = ForesightFactory

if __name__ == '__main__':
    import magicpandas as magic


    class Inner(magic.Frame):

        def conjure(self) -> Self:
            return pd.DataFrame({}, index=[0, 1, 2, 3, 4])

        @magic.column
        def a(self):
            return np.arange(len(self))

        @magic.column
        def b(self):
            return np.arange(len(self))

        # magic is None; add to class' foresights
        @magic.foresight[['a', 'b']]
        def foresight(self):
            ...

        # magic is not None; add to magic's foresights
        @a.foresight
        def a_foresight(self):
            ...

        @b.foresight
        def b_foresight(self):
            """"""


    class Outer(magic.Frame):
        @Inner
        def inner(self):
            ...
"""
write(Inner.foresight)
write(Inner.a_foresight)
write(Inner.b_foresight)
"""
