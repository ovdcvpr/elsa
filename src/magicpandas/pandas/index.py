from __future__ import annotations

from typing import *

import pandas as pd

from magicpandas.magic.cached.cached import cached
from magicpandas.pandas.column import Column

if False:
    from magicpandas import Frame


class Index(
    Column
):
    @cached.diagonal.property
    def __permanent__(self):
        return True

    def __subset__(self, instance, value):
        key = self.__key__
        # todo: I am not sure we are caching a magic.Index instance for the case of MultiIndex
        if key == instance.index.name:
            instance.index = value
        elif key in instance.index.names:
            instance.index = instance.index.set_levels(value, level=key)
        else:
            # column subset
            value = super().__subset__(instance, value)

        return value

    def __subget__(self: Column, outer: Frame, Outer) -> Column:
        key = self.__key__
        owner: Frame = self.__owner__

        if self.__permanent__:
            owner.__permanent__.add(key)
        owner.__columns__.add(key)

        # todo: problem is that accessing while it's a copy will perform setitem
        if key in owner._item_cache:
            # noinspection PyTypeChecker
            # return owner._item_cache[key]
            result = owner._item_cache[key]
            cls = self.__class__
            if (
                isinstance(result, pd.Series)
                and not isinstance(result, cls)
            ):
                result = cls(result)
                owner._item_cache[key] = result
            return result


        # todo: what if key is both in columns and index?
        if key in owner:
            # might already be pd.Series not magic.Column
            try:
                result = owner._get_item_cache(key)
            except TypeError:
                result = owner[key]
            if isinstance(result, Column):
                return result
            if isinstance(result, pd.Series):
                result = self.__subset__(owner, result)
            return result
        elif key in owner.index.names:
            result = owner.index.get_level_values(key)
            return result
        elif self.from_outer:
            func = self.from_outer.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            result = self.__log__(func)
            outer.__inner__ = inner
            result = self.__subset__(owner, result)
        elif self.conjure:
            result = self.__log__(self.conjure)
            result = self.__subset__(owner, result)
        else:
            msg = f'Could not resolve a constructor for {self.__trace__}'
            raise NotImplementedError(msg)

        return result


class index(Index):
    ...

locals()['index'] = Index
