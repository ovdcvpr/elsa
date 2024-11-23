from __future__ import annotations

import pandas as pd
from typing import Self

from functools import cached_property

import magicpandas as magic
from elsa import Prediction


class Maximums(magic.Frame):
    outer: Prediction

    def __repr__(self):
        return super().__repr__()

    @cached_property
    def labels(self) -> Self:
        loc = self.columns.isin(self.outer.label)
        result = self.loc[:, loc]
        return result

    def conjure(self):
        logits = self.outer
        # use the maximum confidence for tokens within a label
        label = (
            logits.confidence
            .T
            .groupby(level='label', sort=False)
            .max()
            .T
        )
        # use the maximum confidence for labels within a cat
        cat = (
            logits.confidence
            .T
            .groupby(level='cat', sort=False)
            .max()
            .T
        )
        concat = label, cat
        # noinspection PyTypeChecker
        result = (
            pd.concat(concat, axis=1)
            .pipe(self.enchant)
        )
        return result
