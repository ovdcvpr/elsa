from __future__ import annotations
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
import magicpandas as magic
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *
from magicpandas.util.weakly import weakly

"""
class Outer:
    @nested.property
    def inner():
        ...

outer.inner
"""


class NestedProperty:
    @weakly.cached_property
    def outer(self) -> Optional[NestedProperty]:
        ...

    @weakly.cached_property
    def Outer(self) -> Optional[NestedProperty]:
        ...

    @property
    def owner(self):
        outer = self.outer
        if isinstance(outer, NestedProperty):
            return outer.owner
        return outer

    def __init__(self, func):
        self.__func__ = func

    def __set_name__(self, owner, name):
        self.__name__ = name
        assert hasattr(owner, '__get__')

    def __get__(self, instance, owner):
        self.outer = instance
        self.Outer = owner
        return self

    def __set__(self, instance, value):
        self.outer = instance
        raise NotImplementedError

    def __delete__(self, instance):
        raise NotImplementedError

    @property
    def nested_name(self):
        return f'{self.outer.__name__}.{self.__name__}'


class nested:
    cached_property = NestedProperty
