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

import pandas as pd
from pandas import Series
from typing import *

from elsa.resource import Resource

if False:
    from elsa import Elsa


class Disjoint(
    Resource,
    magic.Frame
):
    """
    A DataFrame representing all possible reasons that any
    annotation combinations in the dataset may be disjoint. For
    example, "person standing sitting" is disjoint: a person cannot
    be both standing and sitting.
    """
    outer: Elsa

    def conjure(self) -> Self:
        """Called when accessing Elsa.disjoint to instantiate Invalid"""
        elsa = self.outer
        combos = elsa.truth.combos
        unique = combos.outer.unique
        consumed = unique.consumed
        assert not unique.ilabels.difference(consumed.ilabels).any()
        unique = unique.disjoint.all_checks
        consumed = consumed.disjoint.all_checks
        disjoint = unique & consumed
        checks = list(unique.checks)

        docstrings = Series([
            getattr(disjoint.__class__, check).__doc__
            for check in checks
        ], index=checks, name='docstring')

        disjoint.ilabels.unique()
        loc = (
            disjoint
            .loc[:, checks]
            .rename_axis('check', axis=1)
            .stack()
        )
        _ = combos.ilabels
        label = (
            combos
            .drop_duplicates('isyns')
            .set_index('isyns')
            .label
        )
        result: pd.DataFrame = (
            loc
            .loc[loc]
            .reset_index('check')
            [['check']]
            .join(docstrings, on='check')
            .merge(label, left_index=True, right_index=True)
            .sort_index()
            .sort_values('check')
            .loc[:, 'label check docstring'.split()]
        )

        return result
