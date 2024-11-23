from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.scored.cdba.detection import Detection
from elsa.scored.cdba.groups import Groups
from elsa.scored.cdba.has import IMatch
from elsa.scored.cdba.magic import Magic
from elsa.scored.cdba.matches import Matches
from elsa.scored.cdba.stacked import Stacked
from elsa.scored.cdba.summary import Summary
from elsa.scored.scored import Scored
from elsa.scored.cdba.summary import Summary


class Threshold(
    Magic.Blank
):
    @magic.cached.sticky.property
    @magic.portal('alg.txt')
    def score_range(self) -> float:
        """score_thr in the CDBA algorithm"""

    @magic.cached.sticky.property
    @magic.portal('alg.txt')
    def iou(self) -> float:
        """iou_thr in the CDBA algorithm"""


class CDBA(
    Scored,
    IMatch,
):
    outer: Scored
    conjure = False

    @magic.portal('alg.png')
    @magic.portal('alg.txt')
    def __call__(
            self,
            score_range: float = .2,
            iou: float = .80,
            # anchored: bool = False,
            anchored: bool = True,
            *args,
            **kwargs,
    ) -> Self:
        result: Self
        result = self.enchant(self.outer)
        loc = result.ifile.isin(self.root.ifile)
        result = result.loc[loc].copy()
        result.is_anchored = anchored

        # score is used here to select scored
        result.threshold.score_range = score_range

        # iou is used in selecting matches
        magic.portal(Matches.conjure)
        result.threshold.iou = iou

        """
        5:      Compute score range R = max(Scores) − min(Scores)
        6:      if R > score_thr then
        7:          Select bboxes bi where Score(bi) ≥ max(Scores) − score_thr
        8:      else
        9:          Select all bboxes in grp
        """
        loc = result.score_range.values <= result.threshold.score_range
        loc |= result.score.values > result.score_max.values - result.threshold.score_range
        result = result.loc[loc]

        magic.portal('alg.txt')
        magic.portal(Matches.true_positive)
        magic.portal(CDBA.true_positive)
        return result

    @magic.cached.static.property
    def summary(self) -> Summary:
        return self.detection.input.summary

    @magic.column
    def imatch(self):
        _ = self.matches.score

        # matches = self.matches
        # matches.ipred.nunique()
        result = (
            self.matches
            .sort_values('score', ascending=False)
            .reset_index()
            .groupby('ipred', sort=False)
            .imatch
            .first()
            .reindex(self.ipred.values, fill_value=-1)
            .values
        )
        return result

    @magic.test
    def _test_range(self):
        assert self.score_range.max() < self.SCORE_RANGE

    @magic.test
    def _test_subset(self):
        assert not len(self.index.difference(self.scored.index))

    @Stacked
    @magic.portal(Stacked.conjure)
    def stacked(self):
        ...

    @cached_property
    def SCORE_RANGE(self):
        return .2

    @magic.column
    def igroup(self) -> magic[int]:
        result = (
            self.matches.igroup
            .reindex(self.ipred.values, fill_value=-1)
            .values
        )
        loc = result == -1
        result[loc] = np.arange(loc.sum()) + result.max() + 1
        return result

    # @igroup.test
    # def _test_igroup(self):
    #     loc = self.igroup.isin(self.groups.igroup)
    #     assert loc.all(), 'igroup not in groups'

    @magic.column
    def nmatches(self):
        """Determine how many matches each ipred has in the matches"""
        result = (
            self.matches
            .groupby(self.ipred.name)
            .size()
            .reindex(self.ipred, fill_value=0)
            .values
        )
        return result

    @magic.test
    def _test_one_match(self):
        assert self.nmatches.max() <= 1

    def includes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a group contains a label"""
        stacks = self.stacked
        result = (
            stacks
            .includes(label, cat)
            .groupby(stacks.igroup.values)
            .any()
            .reindex(self.igroup.values, fill_value=False)
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a group excludes a label"""
        return ~self.includes(label, cat)

    def get_unique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the igroup and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        stacks = self.stacked
        result = (
            stacks
            .loc[loc]
            .groupby(stacks.igroup.name)
            .ilabel
            .nunique()
            .reindex(self.igroup, fill_value=0)
        )

        return result

    def get_unique_cats(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the igroup and count the number of unique cats
        """
        if loc is None:
            loc = slice(None)
        stacks = self.stacked
        result = (
            stacks
            .loc[loc]
            .groupby(stacks.igroup.name)
            .icat
            .nunique()
            .reindex(self.igroup, fill_value=0)
        )

        return result

    @Groups
    @magic.portal(Groups.conjure)
    def groups(self):
        ...

    @magic.column
    @magic.portal(Groups.conjure)
    def is_groups(self) -> magic[bool]:
        groups = self.groups.any(axis=0)
        igroup = groups.igroup[groups]
        result = self.igroup.isin(igroup)
        return result

    @magic.cached.outer.property
    def scored(self) -> Scored:
        ...

    @Threshold
    def threshold(self):
        ...

    @magic.cached.sticky.property
    def is_anchored(self) -> bool:
        ...

    @magic.column
    def truth_iclass(self):
        result = (
            self.elsa.truth.combos.iclass
            .reindex(self.itruth, fill_value=0)
            .values
        )
        return result

    @Detection
    def detection(self):
        ...

    @magic.column
    def cat(self):
        labels = self.elsa.labels
        _ = labels.cat
        result = (
            labels
            .reset_index()
            .set_index('ilabel')
            .cat
            .loc[self.ilabel]
            .values
        )
        return result

    @Matches
    @magic.portal(Matches.conjure)
    def matches(self):
        ...

    @magic.column
    def true_positive(self) -> magic[bool]:
        result = (
            self.matches.true_positive
            .reindex(self.imatch, fill_value=False)
            .values
        )
        result &= ~self.is_disjoint.values

        return result
