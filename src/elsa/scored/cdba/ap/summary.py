from __future__ import annotations
import pandas as pd

import itertools
from functools import *
from typing import Self

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.scored import summary

if False:
    import elsa.scored.cdba.cdba
    from elsa.classes.classes import Classes
    from elsa.scored.cdba.cdba import CDBA

"""
level
condition
class
iou
score
"""


class Threshold(magic.Magic):
    @cached_property
    def scores(self):
        return np.linspace(80, .5, 96) / 100

    @cached_property
    def ious(self):
        return [.85, .9, .95]


class Summary(
    summary.Summary
):
    outer: CDBA

    def conjure(self) -> Self:
        """
        Compute the AP for each level, condition, and score.
        """
        CDBA = self.outer
        scores = self.thresholds.scores

        it = itertools.product(self.levels, self.condition)
        aps = []
        for level, condition in it:
            cdba = CDBA.subset(level=level, condition=condition)
            for score in scores:
                ap = cdba.ap(score)
                aps.append(ap)

        iterables = self.level, self.condition, scores
        names = 'level', 'condition', 'score'
        frame = (
            pd.MultiIndex
            .from_product(iterables, names=names)
            .to_frame()
            .assign(ap=aps)
            .groupby('level condition score'.split())
            .mean()
            .reset_index()
        )
        return frame

    @cached_property
    def levels(self):
        return ['c', 'cs', 'csa', 'cso', 'csao', None]

    @cached_property
    def condition(self):
        return ['person', 'pair', 'people', None]

    @property
    def iclass(self):
        classes: Classes = self.elsa.classes
        return [*classes.iclass, None]

    @magic.cached.outer.property
    def cdba(self) -> elsa.scored.cdba.cdba.CDBA:
        ...

    @Threshold
    def thresholds(self):
        ...
