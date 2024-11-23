from __future__ import annotations

import math

import pandas as pd

import itertools
from functools import *
from typing import *

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.classes.classes import Classes
    from elsa.annotation.prompts import Prompts


def pad(lst, target_length=3, padding_value=-1):
    return (
            lst
            + (padding_value,)
            * (target_length - len(lst))
    )


class Synonyms(
    Resource,
    magic.Frame
):
    """Create a frame of synonymous prompts, ordered in decreasing "diversity"""
    outer: Classes

    def conjure(self) -> Self:
        classes = self.outer
        try:
            del self.ilabel2isyns
        except AttributeError:
            pass
        try:
            del self.prompts
        except AttributeError:
            pass
        concat = [
            self.function(ilabels)
            for ilabels in classes.ilabels.values
        ]
        repeat = list(map(len, concat))
        ilabels = classes.ilabels.values.repeat(repeat)
        isyns = np.concatenate(concat)
        index = pd.Index(isyns, name='isyns', dtype=self.elsa.prompts.isyns.dtype)
        result = pd.DataFrame({
            'ilabels': ilabels,
        }, index=index)
        assert result.ilabels.isin(self.elsa.truth.ilabels).all()
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

    # def function(self, ilabels: tuple[int, ...]):
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
