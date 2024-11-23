from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import Self, Iterable

from pandas import DataFrame
if False:
    from magicpandas.pandas.series import Series


class SubLoc:
    """
    Index an NDFrame on a level of its MultiIndex.
    This requires that the index level is unique.
    """
    instance: Series | DataFrame = None

    def __get__(self, instance, owner) -> Self:
        if instance is None:
            return None
        result = copy.copy(self)
        result.instance = instance
        return result

    def __getitem__(self, item):
        loc: Series | np.ndarray
        if isinstance(item, tuple):
            loc, cols = item
        else:
            loc = item
            cols = slice(None)
        name = getattr(loc, 'name', None)
        if not name:
            raise ValueError(f'Cannot use {loc} as a key')
        instance = self.instance
        attr = f'{self.__name__}.{name}'
        if attr not in instance.__dict__:
            index = instance.index.get_level_values(name)
            if index.duplicated().any():
                raise ValueError(
                    f'Cannot subloc on {name} because it is not unique'
                )
            data = np.arange(len(index))
            iloc = pd.Series(data, index=index)
            instance.__dict__[attr] = iloc
        else:
            iloc = instance.__dict__[attr]
        iloc = iloc.loc[loc]
        if isinstance(cols, slice):
            result = instance.iloc[iloc]
        elif isinstance(cols, Iterable):
            icol = [instance.columns.get_loc(col) for col in cols]
            result = instance.iloc[iloc, icol]
        else:
            icol = instance.columns.get_loc(cols)
            result = instance.iloc[iloc, icol]
        return result

    def __setitem__(self, item, value):
        loc: Series | np.ndarray
        if isinstance(item, tuple):
            loc, cols = item
        else:
            loc = item
            cols = slice(None)
        name = getattr(loc, 'name', None)
        if not name:
            raise ValueError(f'Cannot use {loc} as a key')
        instance = self.instance
        attr = f'{self.__name__}.{name}'
        if attr not in instance.__dict__:
            index = instance.index.get_level_values(name)
            if index.duplicated().any():
                raise ValueError(
                    f'Cannot subloc on {name} because it is not unique'
                )
            data = np.arange(len(index))
            iloc = pd.Series(data, index=index)
            instance.__dict__[attr] = iloc
        else:
            iloc = instance.__dict__[attr]
        iloc = iloc.loc[loc]
        if isinstance(cols, slice):
            instance.iloc[iloc] = value
        elif isinstance(cols, Iterable):
            icol = [instance.columns.get_loc(col) for col in item[1:]]
            instance.iloc[iloc, icol] = value
        else:
            icol = instance.columns.get_loc(cols)
            instance.iloc[iloc, icol] = value

    def __set_name__(self, owner, name):
        self.__name__ = name
