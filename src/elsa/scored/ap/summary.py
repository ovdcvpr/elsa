from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from functools import *
from typing import *
from typing import Self

import tqdm

import magicpandas as magic
from elsa.evaluation import summary
from elsa.scored.ap.average import Average
from elsa.scored.ap.samples import Samples

if False:
    from elsa.scored.ap.ap import AP


class Row(magic.Series):
    def conjure(self) -> Self:
        return self.outer.loc[self.__name__]


class Summary(summary.Summary):
    outer: AP

    @cached_property
    def averages(self):
        return [
            key
            for key, value in Samples.__dict__.items()
            if isinstance(value, Average)
        ]

    @cached_property
    def samples(self):
        return 'multiclass'.split()

    # def __call__(
    #         self,
    #         method: Literal[
    #             'torch',
    #             'sklearn'
    #         ] = 'torch',
    # ) -> Self:

    # def conjure(self) -> Self:
    def __call__(
            self,
            score_name: str = 'selected.nlse',
    ):
        AVERAGES = self.averages
        PREDICTIONS = self.scored
        PREDICTIONS['iclass prompt'.split()]
        assert len(PREDICTIONS), 'No evaluation data'
        if not score_name.startswith('scores.'):
            score_name = f'scores.{score_name}'
        _ = PREDICTIONS['normw norms norme normn'.split()]
        SCORE = self.__outer__.__name__
        true = np.full(len(PREDICTIONS), True)
        level = PREDICTIONS.level.values
        condition = PREDICTIONS.condition.values
        LEVELS = dict(
            c=level == 'c',
            cs=level == 'cs',
            csa=level == 'csa',
            cso=level == 'cso',
            csao=level == 'csao',
            all_levels=true,
        )
        CONDITIONS = dict(
            person=condition == 'person',
            pair=condition == 'pair',
            people=condition == 'people',
            all_conditions=true,
        )

        it = itertools.product(
            AVERAGES,
            LEVELS.keys(),
            CONDITIONS.keys(),
        )
        names = 'average level condition'.split()
        index = pd.MultiIndex.from_tuples(it, names=names)

        it = itertools.product(
            AVERAGES,
            LEVELS.values(),
            CONDITIONS.values(),
        )
        # todo: parallelize this?
        total = len(AVERAGES) * len(LEVELS) * len(CONDITIONS)
        it = (
            PREDICTIONS
            .loc[level & condition]
            .__getattribute__(SCORE)
            .__getattribute__('multiclass')
            .__getattribute__(average)
            (score_name)
            for average, level, condition, in it
        )
        it = tqdm.tqdm(
            it,
            total=total,
            desc='Computing the mAP Summary...',
            miniters=1
        )

        # EVALUATION.ap.multiclass.macro.torch()
        # score = list(it)
        score = []
        for i in it:
            score.append(i)

        result = (
            pd.concat(score, axis=1, keys=index)
            .replace(-1, np.nan)
            .T
            .assign(score_name=score_name)
        )
        return result

    @magic.cached.static.property
    def macro(self) -> Self:
        loc = self.average == 'macro'
        return self.loc[loc]

    @magic.cached.static.property
    def micro(self) -> Self:
        loc = self.average == 'micro'
        return self.loc[loc]
