from __future__ import annotations
from typing import Self

import magicpandas as magic
from elsa.scored.nms.matches import Matches
from elsa.scored.normal.normal import Normal

"""
Input: preds: predictions
Input: GT: ground-truth
1: pickedPreds = keepPreds = []
2: for k in GT do
3:   for p in preds do
4:     if IoU(p, k) > 0.5 then
5:       pickedPreds = pickedPreds ∪ p
6:     else
7:       keepPreds = keepPreds ∪ p
8:     end if
9:   end for
10:  keepPreds = keepPreds ∪ C-NMS(pickedPreds)
11: end for
12: mAP = AP(keepPreds, GT)
13: return mAP
"""


class NMS(
    Normal,
):
    @magic.portal('alg.png')
    @magic.portal('alg.txt')
    def __call__(
            self,
            iou: float = .80,
            anchored: bool = True,
            *args,
            **kwargs,
    ) -> Self:
        result: Self
        result = self.enchant(self.outer)

        loc = result.ifile.isin(self.root.ifile)
        result = result.loc[loc].copy()

        result.is_anchored = anchored

        """
        4:     if IoU(p, k) > 0.5 then
        5:       pickedPreds = pickedPreds ∪ p
        """
        result.threshold.iou = .5
        matches = result.matches
        result.threshold.iou = iou

        # """10:  keepPreds = keepPreds ∪ C-NMS(pickedPreds) """
        ipred = matches.ipred.loc[matches.keep_preds.values]
        loc = result.ipred.isin(ipred)
        result = result.loc[loc]
        # del result.matches
        # after deleting matches, matches.igroup is not in result.groups
        _ = result.matches
        _ = result.groups

        return result

    @magic.test
    def _test_matches(self):
        assert self.matches.igroup.isin(self.groups.igroup.values).all()
        assert self.matches.igroup.isin(self.igroup.values).all()
        assert self.igroup.isin(self.groups.igroup).all()

    @Matches
    @magic.portal(Matches.nms)
    def matches(self):
        ...
