from __future__ import annotations

from typing import Self

import magicpandas as magic

if False:
    from elsa.prediction.prediction import Prediction


class MaxLogit(magic.Frame):
    outer: Prediction

    @magic.column
    def ifirst(self):
        ...

    @magic.column
    def min(self):
        ...

    @magic.column
    def max(self):
        ...

    def conjure(self) -> Self:
        """
        for each box, we get the max logit and then the max and min
        of these max logits
        """
        logits = self.outer
        index = logits.confidence.columns.get_level_values('ifirst')
        ifirst = (
            logits.confidence
            .set_axis(index, axis=1)
            .idxmax(axis=1)
        )
        min = (
            logits.confidence
            .min(axis=0)
            .reset_index()
            .set_index('ifirst')
            [0]
            .loc[ifirst]
            .values
        )
        max = (
            logits.confidence
            .max(axis=0)
            .reset_index()
            .set_index('ifirst')
            [0]
            .loc[ifirst]
            .values
        )
        result = self.enchant({
            'ifirst': ifirst,
            'min': min,
            'max': max,
        }, index=logits.index)
        return result
