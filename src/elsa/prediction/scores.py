from __future__ import annotations

import math
from typing import *

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

import magicpandas as magic

if False:
    from .prediction import Prediction
    import elsa.prediction.prediction


class Score(magic.column):
    def __set_name__(self, owner: Selection, name):
        super().__set_name__(owner, name)
        owner.scoring.add(name)


class Selection(magic.Magic):
    """
    Before scoring the prediction based on logits, some tokens may
    need to be removed.
    """

    outer: Scores
    scoring = set()

    @magic.cached.outer.property
    def prediction(self) -> elsa.prediction.prediction.Prediction:
        ...

    @Score
    def loglse(self):
        if self.n == 0:
            return np.full(len(self.third), np.nan)
        prediction = self.prediction
        result = logsumexp(prediction.confidence, axis=1)
        result -= math.log(self.n)
        return result

    @Score
    def nlse(self) -> magic[float]:
        """Should be nlse but nlse is in files"""
        try:
            return self.loglse
        except Exception:
            ...
        if self.n == 0:
            return np.full(len(self.third), np.nan)
        prediction = self.prediction
        result = logsumexp(prediction.confidence, axis=1)
        result -= math.log(self.n)
        return result

    @property
    def n(self) -> int:
        return len(self.prediction.confidence.columns)

    @Score
    def argmax(self):
        """max out of the columns for the row"""
        if self.n == 0:
            return np.full(len(self.third), np.nan)
        prediction = self.prediction
        result = prediction.confidence.max(axis=1)
        return result


class Selected(Selection):
    """
    With this selection, we exclude tokens for extraneous particles
    such as "a" or "that".
    """

    @property
    def prediction(self) -> elsa.prediction.prediction.Prediction:
        result = super().prediction.without_extraneous_tokens
        # if not len(result.columns):
        #     raise ValueError('No columns in prediction')
        return result


class Scores(magic.Frame.Blank):
    outer: Prediction

    @Selection
    def whole(self):
        """ The whole prediction, including extraneous tokens. """

    @Selected
    def selected(self):
        """ The prediction with extraneous tokens removed. """

    @property
    def everything(self):
        # todo: this is terribly slow for what it is
        whole = self.whole
        selected = self.selected
        keys = whole.scoring
        for key in keys:
            getattr(whole, key)
        for key in keys:
            getattr(selected, key)
        return self

    @magic.frame
    def summaries(self):
        def summary(series):
            return pd.Series({
                'count': series.count(),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                '25%': series.quantile(0.25),
                '50%': series.median(),
                '75%': series.quantile(0.75),
                'max': series.max()
            })

        summaries = {
            key: summary(self[key])
            for key in self.columns
        }
        result = pd.DataFrame(summaries)
        return result

    def conjure(self) -> Self:
        index = self.outer.index
        result = self.enchant(index=index)
        return result
