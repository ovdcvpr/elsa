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
from elsa.scored.cdba import has

from typing import Self

import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.scored.cdba.magic import Magic

if False:
    from elsa.scored.cdba.cdba import CDBA


class Stacked(
    has.IPred,
    Magic,
):
    ...

    def conjure(self) -> Self:
        cdba = self.cdba
        stacked = self.elsa.classes.stacked
        level = stacked.ilabels.name
        size = (
            stacked
            .groupby(level=level, sort=False, observed=True)
            .size()
            .loc[cdba.ilabels]
            .values
        )
        ipred = cdba.ipred.repeat(size).values
        igroup = cdba.igroup.repeat(size).values
        index = cdba.ipred.name
        result = (
            self.elsa.classes.stacked
            .loc[self.cdba.ilabels]
            .reset_index()
            .assign(ipred=ipred, igroup=igroup )
            .set_index(index)
        )
        return result

    @magic.column
    def imatch(self):
        """truth box that the combo was matched to"""

    # @magic.index
    @magic.column
    def ibox(self):
        """truth box that the combo was matched to"""

    # @magic.index
    @magic.column
    def ilabel(self):
        """synonym ID of the label """

    # @magic.index
    @magic.column
    def ilabels(self):
        ...

    def includes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a match contains a label"""
        if label and cat:
            raise ValueError('label and cat cannot both be provided')
        if label is not None:
            ilabel = self.synonyms.ilabel.from_label(label)
            loc = self.ilabel == ilabel
        elif cat is not None:
            loc = self.cat == cat
        else:
            raise ValueError('label or cat must be provided')

        # noinspection PyTypeChecker
        return loc

    def excludes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a match excludes a label"""
        return ~self.includes(label, cat)


    # def get_nunique_labels(self, loc=None) -> Series[bool]:
    #     """
    #     Pass a mask that is aligned with the annotations;
    #     group that mask by the ibox and count the number of unique labels
    #     """
    #     if loc is None:
    #         loc = slice(None)
    #     result = (
    #         self.ilabel
    #         .loc[loc]
    #         .groupby(level='ilabels', observed=True)
    #         .nunique()
    #     )
    #     return result

    def get_nunique(
            self,
            loc=None,
            nunique: str = 'ilabel',
    ):
        """
        Given a mask that is aligned with the annotations,
        determine the number of unique labels in each combo
        """
        if loc is None:
            loc = slice(None)
        getattr(self, nunique)
        igroup = (
            self.igroup
            .pipe(pd.Index)
            .unique()
        )
        by = igroup.name

        # result = (
        #     getattr(self, nunique)
        #     .loc[loc]
        #     .groupby(by, observed=True, sort=False)
        #     .nunique()
        #     .reindex(igroup, fill_value=0)
        # )
        result = (
            self
            .loc[loc]
            .reset_index()
            .groupby(by, observed=True, sort=False)
            .__getattr__(nunique)
            .nunique()
            .reindex(igroup, fill_value=0)
        )
        return result



    @magic.column
    def cat(self) -> magic[str]:
        result = (
            self.elsa.labels.cat
            .indexed_on(self.ilabel)
            .values
        )
        return result

    @magic.test
    def _test_length(self):
        expected = sum(map(len, self.outer.ilabels.values))
        actual = len(self)
        assert actual == expected

    @magic.index
    def ipred(self) -> magic[int]:
        ...

    @magic.column
    def igroup(self):
        result = (
            self.cdba.igroup
            .indexed_on(self.ipred.values)
            .values
        )
        return result

    @magic.column
    def ngroup(self):
        result = (
            self.cdba.ngroup
            .indexed_on(self.ipred.values)
            .values
        )
        return result


    def __align__(self, owner: CDBA = None) -> Self:
        loc = self.ipred.isin(owner.ipred)
        result = self.loc[loc]
        return result


