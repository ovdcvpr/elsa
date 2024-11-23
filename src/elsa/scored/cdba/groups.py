from __future__ import annotations

import pandas as pd
from numpy import ndarray
from pandas import Series
from typing import *

import magicpandas as magic
from elsa.resource import Resource
from elsa.scored.cdba.magic import Magic
from elsa.scored.cdba.matches import Matches
from magicpandas.pandas.ndframe import NDFrame

if False:
    from elsa.scored.cdba.cdba import CDBA
    from elsa.scored.cdba.matches import Matches
    import elsa.scored.cdba.cdba


class Check(magic.column):
    """Column is being used as a check for disjoint combinations"""


check = Union[Check, Series, ndarray]
globals()['check'] = Check


class Groups(
    magic.Frame,
    Magic,
):
    """
    Computes whether groups of matches are disjoint. This is required
    for determining which matches are true positives.
    """

    outer: CDBA

    @magic.portal(Matches.true_positive)
    def conjure(self) -> Self:
        index = (
            self.cdba.igroup
            .pipe(pd.Index)
            .unique()
        )
        result = self.enchant(index=index)
        return result


    @magic.column
    def is_disjoint(self):
        disjoint = self.matches.disjoint
        threshold = disjoint.threshold
        groupby = disjoint.is_disjoint.groupby(disjoint.igroup)
        result = (
            groupby.sum()
            .div(groupby.size())
            .ge(threshold)
            .reindex(self.igroup, fill_value=False)
            .values
        )

        return result

    @magic.index
    def igroup(self) -> magic[int]:
        ...

    @magic.column
    def ngroup(self) -> magic[int]:
        result = (
            self.cdba
            .groupby(self.cdba.igroup.name)
            .size()
            .reindex(self.igroup, fill_value=0)
            .values
        )
        return result

    @magic.column
    def ntrue(self):
        cdba = self.cdba
        _ = cdba['igroup true_positive'.split()]
        result = (
            cdba
            .groupby('igroup')
            .true_positive
            .sum()
            .reindex(self.igroup, fill_value=0)
            .values
        )
        return result

    @magic.column
    def score_range(self):
        """range of the scores at the same itruth"""
        cdba = self.cdba
        max = (
            cdba
            .groupby(cdba.igroup.name)
            .score
            .max()
        )
        min = (
            cdba
            .groupby(cdba.igroup.name)
            .score
            .min()
        )
        result = max - min
        return result

    @magic.column
    def score_max(self):
        """max of the scores at the same itruth"""
        cdba = self.cdba
        result = (
            cdba
            .groupby(cdba.igroup.name)
            .score
            .max()
        )
        return result

    # def __align__(self, owner: NDFrame = None) -> Self:
    #     loc = self.igroup.isin(self.cdba.igroup)
    #     result = self.loc[loc]
    #     return result
