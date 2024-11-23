from __future__ import annotations

import itertools
from functools import *
from typing import *

import numpy as np
from numba import njit

import magicpandas as magic
from elsa.scored.normal import matches


@njit
def non_maximum_suppression_njit(group: np.ndarray, threshold: float) -> np.ndarray:
    x1 = group[:, 0].astype(np.float32)
    y1 = group[:, 1].astype(np.float32)
    x2 = group[:, 2].astype(np.float32)
    y2 = group[:, 3].astype(np.float32)
    score = group[:, 4].astype(np.float32)
    imatch = group[:, 5].astype(np.int64)

    areas = (x2 - x1 + 1).astype(np.float32) * (y2 - y1 + 1).astype(np.float32)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]]).astype(np.float32)
        yy1 = np.maximum(y1[i], y1[order[1:]]).astype(np.float32)
        xx2 = np.minimum(x2[i], x2[order[1:]]).astype(np.float32)
        yy2 = np.minimum(y2[i], y2[order[1:]]).astype(np.float32)

        w = np.maximum(0, xx2 - xx1 + 1).astype(np.float32)
        h = np.maximum(0, yy2 - yy1 + 1).astype(np.float32)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    keep_array = np.array(keep, dtype=np.int64)
    result = imatch[keep_array]
    return result


class Matches(matches.Matches):
    @magic.column
    @magic.portal('alg.txt')
    def picked_preds(self):
        result = (
            self.iou
            .gt(.5)
            .groupby(self.ipred.values)
            .any()
            .loc[self.ipred.values]
            .values
        )
        return result

    @magic.column
    @magic.portal('alg.txt')
    def keep_preds(self):
        loc = self.picked_preds
        result = ~loc
        nms = self.loc[loc].nms
        result |= self.ipred.isin(nms.ipred.values)
        return result

    @cached_property
    def nms_threshold(self):
        return .5

    @magic.cached.static.property
    def nms(self) -> Self:
        _ = self['score']
        _ = self['itruth normw norms norme normn score'.split()].values
        MATCHES = self.sort_values('score', ascending=False)
        grouped = self.groupby('itruth', sort=False)
        loc = (
            grouped
            .size()
            .eq(1)
            .loc[self.itruth.values]
            .values
        )
        imatch = self.imatch.values[loc]
        self = self.loc[~loc]
        grouped = self.groupby('itruth', sort=False)

        ARRAY = self.reset_index()[['normw', 'norms', 'norme', 'normn', 'score', 'imatch']].values
        arrays = (
            ARRAY[loc]
            for loc in grouped.indices.values()
        )

        it = map(non_maximum_suppression_njit, arrays, itertools.repeat(self.nms_threshold))
        appendix = np.concatenate(list(it))

        # Append the single match cases if needed
        imatch = np.concatenate((imatch, appendix))
        result = MATCHES.loc[imatch]
        return result
