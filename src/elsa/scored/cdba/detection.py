from __future__ import annotations

from functools import *
from typing import *

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.classes.has import ICls
from elsa.scored.cdba.magic import Magic

if False:
    from elsa.scored.cdba.cdba import CDBA
    import elsa.scored.cdba.ap.ap
    import elsa.scored.cdba.summary


class Annotation(
    ICls,
    # magic.Frame,
):
    outer: Input

    @magic.index
    def iclass(self) -> magic[int]:
        ...


class Threshold(magic.Magic):
    @magic.index
    def iou(self):
        ...

    @magic.index
    def conf(self):
        ...

    # def __call__(
    #         self,
    #         iou: list[float],
    #         conf: list[float]
    # ) -> Input:
    #     ...

    @cached_property
    def scores(self):
        return np.arange(.3, 1.0, .05)

    @cached_property
    def ious(self):
        return np.arange(.8, 1.0, .05)


class Input(
    magic.Frame,
    Magic
):
    @magic.column
    def tp(self):
        ...

    @magic.column
    def conf(self):
        ...

    @Annotation
    def pred(self):
        ...

    @Annotation
    def target(self):
        ...

    @magic.column
    def iou(self):
        ...

    @Threshold
    def threshold(self) -> Self:
        ...

    @Threshold
    def threshold(self) -> Self:
        ...

    @magic.column
    def ipred(self):
        ...

    # @AP
    # @magic.portal(AP.__call__)
    # def ap(self):
    #     ...

    # ap: elsa.scored.cdba.ap.ap.AP
    # if False:
    #     ap = elsa.scored.cdba.ap.ap.AP()
    #
    # @magic.delayed
    # def ap(self) -> elsa.scored.cdba.ap.ap.AP:
    #     ...

    # ap = magic.delayed()

    # ap: elsa.scored.cdba.ap.ap.AP = magic.delayed()
    # @magic.delayed
    # def ap(self) -> elsa.scored.cdba.ap.ap.AP:
    #     ...

    summary: elsa.scored.cdba.summary.Summary

    if False:
        summary = elsa.scored.cdba.summary.Summary()

    @magic.delayed
    def summary(self) -> elsa.scored.cdba.summary.Summary:
        ...


class Matched(
    Input
):
    # @magic.portal(Truth.unmatched)
    def conjure(self) -> Self:
        """
        ultralytic's ap_per_class computation takes four parameters:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            conf (np.ndarray): Array of confidence scores of the detections.
            pred_cls (np.ndarray): Array of predicted classes of the detections.
            target_cls (np.ndarray): Array of true classes of the detections.

        These four arrays are aligned, which means if it is only represented by
        matches, there will be missing false positives and false negatives.
        For this, we must add additional entries to the arrays to include
        false positives and false negatives.

        False negatives are represented as:
            tp = 0
            conf = 1.
            pred_cls = 0
            target_cls != 0

        False positives are represented as:
            tp = 0
            conf != 0
            pred_cls != 0
            target_cls = 0

        """

        cdba = self.cdba
        loc = cdba.ifile.isin(self.root.ifile)
        cdba = cdba.loc[loc].copy()
        tp = cdba.true_positive.values
        conf = cdba.score.values
        pred_cls = cdba.iclass.values
        target_cls = cdba.truth_iclass.values
        ipred = cdba.ipred.values
        iou = cdba.iou.values

        result = pd.DataFrame({
            'tp': tp,
            'conf': conf,
            'pred.iclass': pred_cls,
            'target.iclass': target_cls,
            'ipred': ipred,
            'iou': iou
        })

        return result


class Unmatched(
    Input
):
    def conjure(self) -> Self:
        # todo: which truth annotations count as false negatives when we subset?
        truth = self.elsa.truth.combos
        cdba = self.cdba
        loc = cdba.ifile.isin(self.root.ifile)
        cdba = cdba.loc[loc].copy()
        loc = ~truth.ibox.isin(cdba.itruth.values)
        target_cls = truth.iclass.loc[loc].values
        iou = 1.
        ipred = -1
        result = pd.DataFrame({
            'tp': False,
            'conf': 0.,
            'pred.iclass': 0,
            'target.iclass': target_cls,
            'iou': iou,
            'ipred': ipred
        })
        return result


class Detection(
    Magic
):
    outer: CDBA

    @Unmatched
    @magic.portal(Unmatched.conjure)
    def unmatched(self):
        ...

    @Matched
    @magic.portal(Matched.conjure)
    def matched(self):
        ...

    @magic.cached.sticky.property
    def nclasses(self) -> Optional[int]:
        return None

    """
    # todo
    sometimes, everything is 
    """

    @Input
    @magic.portal(Matched.conjure)
    def input(self):
        # nunique for cdba
        objs = [self.matched, self.unmatched]
        result = (
            pd.concat(objs, ignore_index=True)
            .pipe(Input)
        )
        return result

    # @input.test
    # def _test_subset(self):
    #     level = self.outer.subset.level
    #     condition = self.outer.subset.condition
    #     if level != 'any':
    #         unique = set(self.input.pred.level.unique())
    #         unique.remove('')
    #         if unique:
    #             assert level in unique, f'{level} not in {unique}'
    #             assert len(unique) == 1, f'{unique} not 1'
    #     if condition != 'any':
    #         unique = set(self.input.pred.condition.unique())
    #         unique.remove('')
    #         if unique:
    #             assert condition in unique, f'{condition} not in {unique}'
    #             assert len(unique) == 1, f'{unique} not 1'
