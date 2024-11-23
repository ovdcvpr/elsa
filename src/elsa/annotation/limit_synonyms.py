from __future__ import annotations

import itertools
import math
from functools import *
from itertools import *
from typing import *

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.annotation.prompts import Prompts


def pad(lst, target_length=3, padding_value=-1):
    return (
            lst
            # + (padding_value,)
            + [padding_value]
            * (target_length - len(lst))
    )


class LimitSynonyms(Prompts):
    """
    Limit the number of synonymous prompts to a certain number.
    For example:
        a person walking
        a person strolling
        an individual walking
        an individual strolling

    There may be much more prompts available by default than the user
    requires, so there is a need to limit the amount of synonymous prompts.
    However, it is not sufficient to simply take the first N prompts.
    For example, selecting the first two prompts would result in:
        a person walking
        a person strolling

    However, these are still quite similar and do not provide a diverse
    set of prompts. Ideally we would like to select the most diverse
    set of prompts. In this case it would be:
        a person walking
        an individual strolling

    In this module, we create a subset of the prompts by selecting the
    first prompt for each set of synonymous prompts, and then progressively
    selecting the prompt that has the most unused synonymous labels.
    """
    outer: Prompts

    def __call__(self, n: int, diverse=True) -> Self:
        prompts = self.outer
        synonyms = self.elsa.classes.synonyms
        loc = synonyms.isyns.isin(prompts.isyns)
        isyns = (
            synonyms
            .loc[loc]
            .groupby('ilabels', sort=False, observed=True)
            .head(n)
            .isyns
        )
        result = prompts.indexed_on(isyns)
        assert np.all(result.isyns.values == isyns.values)
        loc  = result.ilabels == (0,5)
        result.loc[loc]
        loc = synonyms.ilabels == (0,5)
        loc = prompts.isyns == (0,119)
        prompts.loc[loc]
        loc = prompts.isyns == (1,120)
        prompts.loc[loc]

        synonyms.loc[loc]
        return result

    def __call__(self, n: int) -> Self:
        try:
            del self.ilabel2isyns
        except AttributeError:
            pass
        try:
            del self.prompts
        except AttributeError:
            pass
        prompts = self.outer
        ilabels = prompts.ilabels.unique()
        # isyns = set(prompts.isyns.values)
        isyns = set(chain.from_iterable(prompts.isyns.values))
        concat = [
            self.function(ilabels, isyns)
            for ilabels in ilabels
        ]
        repeat = list(map(len, concat))
        ilabels = ilabels.repeat(repeat)
        isyns = np.concatenate(concat)
        index = pd.Index(isyns, name='isyns', dtype=self.elsa.prompts.isyns.dtype)
        limit = pd.DataFrame({
            'ilabels': ilabels,
        }, index=index)
        assert limit.ilabels.isin(self.elsa.truth.ilabels).all()

        result: Prompts = (
            prompts
            .reset_index()
            .groupby('ilabels', sort=False, observed=True)
            .head(n)
        )
        return result

    @cached_property
    def ilabel2isyns(self):
        return (
            self.elsa.synonyms
            .groupby('ilabel', observed=True)
            .isyn
            .apply(tuple)
            .to_dict()
        )

    @cached_property
    def prompts(self) -> Prompts:
        return (
            self.elsa.prompts
            .reset_index()
            .set_index('ilabels')
        )


    def function(
            self,
            ilabels: tuple[int, ...],
            isyns: set[int],
    ):
        ilabels2isyns: dict[int, list] = {
            ilabel: [
                isyn
                for isyn in self.ilabel2isyns[ilabel]
                if isyn in isyns
            ]
            for ilabel in ilabels
        }

        alternative = self.prompts.isyns.loc[[ilabels]]

        nsynonyms = max(map(len, ilabels2isyns.values()))
        nlabels = len(ilabels2isyns)
        ntuples = nsynonyms ** nlabels
        nints = nlabels * nsynonyms
        indices = np.arange(ntuples)
        ISYNS = np.fromiter((
            isyn
            for isyns in ilabels2isyns.values()
            for isyn in pad(isyns, nsynonyms)
        ), dtype=int, count=nints)
        reps = ntuples // nsynonyms
        tile = [
            np.tile(indices[i * nsynonyms:(i + 1) * nsynonyms], reps)
            for i in range(nlabels)
        ]
        iteration = np.c_[tile].T

        ifirst = np.arange(0, nints, nsynonyms)

        it = itertools.product(range(nsynonyms), repeat=nlabels)
        carryover = (
            np.array(list(it))
            .repeat(nsynonyms, axis=0)
            [:len(iteration)]
        )

        iloc = iteration + carryover
        iloc %= nsynonyms
        iloc += ifirst
        assert np.all(iloc.min(axis=0) >= ifirst)
        assert np.all(iloc.max(axis=0) < ifirst + nsynonyms)
        assert len(np.unique(iloc, axis=0)) == ntuples
        isyns = ISYNS[iloc]

        loc = np.all(isyns != -1, axis=1)
        isyns = isyns[loc]
        result = np.fromiter(map(tuple, isyns), dtype=object, count=len(isyns))

        lengths = list(map(len, ilabels2isyns.values()))
        total = math.prod(lengths)
        assert len(result) == total

        # why are we getting 0, 1, 2,3 in isyns[:, 2]?
        diff = pd.Index(alternative).difference(result)
        assert not len(diff)
        return result

    @magic.index
    def isyns(self) -> magic[tuple[int]]:
        ...

    @magic.column
    def ilabels(self) -> magic[tuple[int]]:
        ...

    @magic.column
    def _isyns_in_ilabels(self):
        isyns2ilabels = (
            self.elsa.synonyms
            .reset_index()
            .set_index('isyn')
            .ilabel
            .to_dict()
        )
        it = zip(self.ilabels.values, self.isyns.values)
        result = [
            all(
                isyns2ilabels[isyn] in ilabels
                for isyn in isyns
            )
            for ilabels, isyns in it
        ]
        return result
