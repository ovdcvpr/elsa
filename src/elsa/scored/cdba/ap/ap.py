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

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.scored.cdba.detection import Input

if False:
    from elsa.scored.cdba.detection import Input


class AP(
    Input
):
    outer: Input
    # def __call__(
    #         self,
    #         score: float,
    # ) -> tuple[int, int, int]:
    #     """Calculate AP at a given score."""
    #     cdba = self.outer
    #     loc = cdba.score.values >= score
    #     tp = cdba.true_positive.loc[loc].values
    #     fn = cdba.loc[loc].false_negative.values
    #     pred_cls = cdba.iclass.values
    #     target_cls = cdba.truth.iclass.values
    #
    #     tp, fp, p, r, f1, ap = ap_per_class(tp, fn, pred_cls, target_cls)
    #     return ap
    #
    #
    # @Summary
    # @magic.portal(Summary.conjure)
    # def summary(self):
    #     ...


    def __call__(
            self,
            ious: list[float],
            confs: list[float],
    ):
        frame = self.outer.sort_values('conf', ascending=False)
        # in recall accumulation, denominator is fixed
        nclass = (
            frame
            .groupby('target.iclass')
            .size()
            .loc[frame.target.iclass.values]
            .values
        )
        frame['denominator'] = nclass

        # broadcast each threshold conf to frame
        loc = frame.pred.iclass.values != 0
        frame = frame.loc[loc]
        iloc = np.arange(len(frame)).repeat(len(confs))
        conf = np.array(confs).repeat(len(frame))
        frame = (
            frame
            .iloc[iloc]
            .assign(**{'threshold.conf': conf, })
        )

        # broadcast each threshold iou to frame
        iloc = np.arange(len(frame)).repeat(len(ious))
        iou = np.array(ious).repeat(len(frame))
        frame = (
            frame
            .iloc[iloc]
            .assign(**{'threshold.iou': iou, })
        )
        groupby = 'threshold.iou threshold.conf target.iclass'.split()
        sort = groupby + ['conf']

        # threshold for each group
        loc = frame.iou.values >= frame.threshold.iou.values
        loc &= frame.conf.values >= frame.threshold.conf.values
        frame = frame.loc[loc]

        size = (
            frame
            .reset_index()
            .groupby(groupby, sort=False)
            .size()
            .values
        )
        ilast = np.cumsum(size) - 1

        # accumulate tp for each group
        tp = np.cumsum(frame.tp.values)
        start = tp[ilast[:-1]]
        sub = np.r_[0, start].repeat(size)
        tp -= sub

        # accumulate fp for each group
        fp = np.cumsum(~frame.tp.values)
        start = fp[ilast[:-1]]
        sub = np.r_[0, start].repeat(size)
        fp -= sub

        # check for proper accumulation
        loc = tp == 0
        loc &= fp == 0
        assert not loc.any()

        precision = tp / (tp + fp)
        recall = tp / frame.denominator.values

        frame: Self = (
            frame
            .assign(precision=precision, recall=recall)
            .reset_index()
            .sort_values(sort + ['recall'])
        )

        # trapz is deprecated but we use old numpy
        try:
            trap = np.trapezoid
        except Exception:
            # noinspection PyUnresolvedReferences
            trap = np.trapz

        def apply(frame: Self):
            # Precision must be monotonically decreasing
            precision = np.maximum.accumulate(
                frame.precision.values[::-1],
            )[::-1]
            # Compute AP using the trapezoidal rule
            recall = frame.recall.values
            ap = trap(precision, recall)
            return ap

        ap = (
            frame
            .groupby(groupby, sort=False)
            .apply(apply)
        )
        return ap



