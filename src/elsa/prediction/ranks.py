from __future__ import annotations

import math
import numpy as np
from functools import *
from numpy import ndarray
from pandas import Series
from scipy.special import logsumexp
from typing import *

import magicpandas as magic

if False:
    from .prediction import Prediction


class Rank(magic.Magic):
    outer: Ranks

    @property
    def logits(self):
        return self.outer.outer

    @staticmethod
    def apply(result):
        iloc = np.argsort(result)[::-1]
        rank = np.arange(len(result))
        result = Series(rank, index=iloc).loc[rank].values
        return result

    def rank(self, scores: ndarray | Series) -> ndarray:
        logits = self.logits
        result = np.full(len(scores), -1, dtype=int)
        SCORES = scores
        ilocs = (
            logits
            .groupby(logits.file, sort=False)
            .indices.values()
        )
        for iloc in ilocs:
            scores = SCORES[iloc]
            i = np.argsort(scores)[::-1]
            rank = np.arange(len(scores))
            assign = (
                Series(rank, index=i)
                .loc[rank]
                .values
            )
            result[iloc] = assign
        assert (result >= 0).all()
        return result

    @magic.column
    def lse(self) -> magic[float]:
        logits = self.logits
        result = logsumexp(logits.confidence, axis=1)
        rank = self.rank(result)
        return rank

    @magic.column
    def argmax(self):
        logits = self.logits
        result = logits.confidence.max(axis=1)
        rank = self.rank(result)
        return rank

    @magic.column
    def nlse(self) -> magic[float]:
        logits = self.logits
        result = logsumexp(logits.confidence, axis=1)
        result -= math.log(self.n)
        rank = self.rank(result)
        return rank

    # @magic.column
    # def avglse(self) -> magic[float]:
    #     logits = self.logits
    #     result = logsumexp(logits.confidence * self.n, axis=1)
    #     result /= self.n
    #     rank = self.rank(result)
    #     return rank
    #

    @property
    def n(self) -> int:
        return len(self.logits.confidence.columns)


class Selected(Rank):

    @property
    def logits(self):
        return super().logits.without_extraneous_tokens


class Ranks(magic.Frame):
    outer: Prediction
    whole = Rank()
    selected = Selected()

    def conjure(self) -> Self:
        index = self.outer.index
        result = self.enchant(index=index)
        return result
