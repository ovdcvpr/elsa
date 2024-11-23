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
from elsa.resource import Resource

# # class Summary(
#     Resource,
#     magic.Frame
# ):
class Summary(
    Resource,
    magic.Frame
):

    @magic.index
    def average(self):
        ...

    @magic.index
    def level(self):
        ...

    @magic.index
    def condition(self):
        ...

    @magic.index
    def algorithm(self):
        ...

    @magic.index
    def score_name(self):
        ...

    @magic.cached.static.property
    def all_conditions(self) -> Self:
        loc = self.condition == 'all_conditions'
        return self.loc[loc]

    @magic.cached.static.property
    def nms(self):
        loc = self.algorithm == 'nms'
        return self.loc[loc]

    @magic.cached.static.property
    def cdba(self):
        loc = self.algorithm == 'cdba'
        return self.loc[loc]

    @magic.cached.static.property
    def no_algorithm(self):
        loc = self.algorithm == 'no_algorithm'
        return self.loc[loc]

    @magic.cached.outer.property
    def eval(self) -> elsa.evaluation.evaluation.Evaluation:
        ...

    @magic.cached.outer.property
    def scored(self) -> elsa.scored.scored.Scored:
        ...

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...

    @magic.column
    def map(self):
        ...

    @magic.column
    def map_50(self):
        ...

    @magic.column
    def map_75(self):
        ...

    @magic.column
    def map_small(self):
        ...

    @magic.column
    def map_medium(self):
        ...

    @magic.column
    def map_large(self):
        ...

    @magic.column
    def mar_100(self):
        ...

    @magic.column
    def mar_1000(self):
        ...

    @magic.column
    def mar_10000(self):
        ...

    @magic.column
    def mar_small(self):
        ...

    @magic.column
    def mar_medium(self):
        ...

    @magic.column
    def mar_large(self):
        ...

    @magic.column
    def map_per_class(self):
        ...

    @magic.column
    def mar_10000_per_class(self):
        ...
