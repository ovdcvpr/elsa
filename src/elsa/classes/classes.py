from __future__ import annotations

from functools import *

import numpy as np
import pandas as pd
from pandas import Series
from typing import *

import magicpandas as magic
from elsa.classes.stacked import Stacked
from elsa.resource import Resource
from elsa.classes.disjoint import Disjoint
from elsa.classes.synonyms import Synonyms
from elsa.classes.subclasses import SubClasses
from elsa.classes.nunique import NUnique
from elsa.classes.labels import Labels


class Classes(
    Resource,
    magic.Frame
):
    """
    Root instance for the project, encapsulating all resources into
    one file for easy access and manipulation.
    """
    @SubClasses
    @magic.portal(SubClasses.conjure)
    def subclasses(self):
        ...

    @Labels
    @magic.portal(Labels.conjure)
    def labels(self):
        ...

    @Stacked
    @magic.portal(Stacked.conjure)
    def stacked(self):
        ...

    @Disjoint
    @magic.portal(Disjoint.conjure)
    def disjoint(self):
        ...

    @Synonyms
    @magic.portal(Synonyms.conjure)
    def synonyms(self):
        ...

    @NUnique
    @magic.portal(NUnique.conjure)
    def nunique_per_class(self):
        ...

    @magic.index
    def ilabels(self) -> magic[tuple[int]]:
        """ilabels index"""

    def conjure(self) -> Self:
        truth = self.elsa.truth.combos
        ilabels = pd.Categorical(truth.ilabels.cat.categories)
        result = pd.DataFrame({
            'ilabels': ilabels,
        }).set_index('ilabels')
        return result

    @magic.frame
    def subcombo(self):
        iloc = np.arange(len(self))
        arrays = iloc, iloc
        names = 'ileft iright'.split()
        index = pd.MultiIndex.from_product(arrays, names=names)
        ileft = index.get_level_values('ileft')
        iright = index.get_level_values('iright')
        left = self.iloc[ileft]
        right = self.iloc[iright]

        data = (
            np.any(left.values & ~right.values, axis=1)
            .__invert__()
            .astype(np.int8)
        )
        arrays = left.index, right.index
        names = 'ilabels ilabels'.split()
        index = pd.MultiIndex.from_arrays(arrays, names=names)
        result = (
            Series(data, index=index)
            .unstack()
        )
        return result

    @magic.series
    def is_subcombo(self) -> magic[bool]:
        labels = self.labels
        arrays = labels.ilabels.values, labels.ilabels.values
        index = pd.MultiIndex.from_product(arrays)
        ileft = index.get_level_values(0)
        iright = index.get_level_values(1)
        left = labels.loc[ileft]
        right = labels.loc[iright]

        data = ~np.any(left.values & ~right.values, axis=1)
        result = Series(data, index=index)
        return result

    @magic.column
    def iclass(self) -> Series[int]:
        """
        assign an integer to each class or ilabels tuple;
        0 is reserved for the background class
        """
        return np.arange(len(self))

    @magic.series
    def ilabels_string(self) -> Series[str]:
        """
        Tuples are not easily serialized; store them as strings instead.
        """
        ilabels = self.ilabels

        def func(ilabels: tuple[int]):
            return ' '.join(map(str, ilabels))

        strings = ilabels.map(func)
        result = Series(strings, index=ilabels, dtype='category')
        return result

    @magic.column.from_options(dtype='category')
    # @magic.column
    def label(self) -> magic[str]:
        """String label for each annotation assigned by the dataset"""
        stacked = self.stacked
        icls = stacked.iclass.name
        ilabel = stacked.ilabel.name
        _ = stacked.label
        by = [icls, ilabel]
        label: Series = (
            stacked
            .sort_values(by=by)
            .groupby(icls, sort=False)
            .label
            .apply(' '.join)
        )
        categories = label.unique().tolist()
        categories.append('')
        categories = pd.CategoricalDtype(categories)
        label = (
            label
            .astype(categories)
            .reindex(self.iclass, fill_value='')
            .values
        )
        return label

    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        if label and cat:
            raise ValueError('label and cat cannot both be provided')
        stacked = self.stacked
        if label is not None:
            ilabel = self.elsa.synonyms.ilabel.from_label(label)
            loc = stacked.ilabel == ilabel

        elif cat is not None:
            loc = stacked.cat == cat
            loc |= stacked.cat_char.values == cat
        else:
            raise ValueError('label or cat must be provided')
        result = (
            Series(loc)
            .groupby(stacked.ilabels.values, observed=True)
            .any()
            .reindex(self.ilabels.values, fill_value=False)
        )
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        return ~self.includes(label, cat)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        result = (
            self.stacked.ilabel
            .loc[loc]
            .groupby(level='ilabels', observed=True)
            .nunique()
            .reindex(self.ilabels, fill_value=0)
        )
        return result

    @magic.column
    def is_disjoint(self) -> magic[bool]:
        # self.disjoint.
        result = (
            self.disjoint
            .any(axis=1)
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def condition(self) -> magic[Literal['person', 'pair', 'people',]]:
        """
        The mutually exclusive value for the condition category within
        a combo:
            person, pair, or people
        """
        result = Series('', index=self.index)
        loc = self.includes(label='person')
        result.loc[loc] = 'person'
        loc = self.includes(label='pair')
        result.loc[loc] = 'pair'
        loc = self.includes(label='people')
        result.loc[loc] = 'people'
        return result

    @magic.column.from_options(dtype='category')
    def level(self) -> magic[str]:
        """
        The level of the combo, or the characters, e.g. cs, csa, csaoa

        c: condition
        cs: condition, state
        csa: condition, state, activity
        cso: condition, state, others
        csao: condition, state, activity, others
        """
        # todo: must we require cs if a? must we require csa if o?
        includes = self.includes
        loc = includes(cat='condition')
        condition = np.where(loc, 'c', '')
        loc = includes(cat='state')
        state = np.where(loc, 's', '')
        loc = includes(cat='activity')
        activity = np.where(loc, 'a', '')
        loc = includes(cat='others')
        others = np.where(loc, 'o', '')
        sequence = condition, state, activity, others
        result = reduce(np.core.defchararray.add, sequence)

        # categories = ['', *'c cs csa cso csao'.split()]
        # categories =';c;cs;csa;cso;csao'.split(';')
        # categories = pd.CategoricalDtype(categories)
        result = Series(result, index=self.index, dtype='category')
        return result

    @magic.column
    def frequency(self) -> magic[int]:
        """
        The number of times each class was represented in the ground
        truth combos.
        """
        combos = self.truth.combos
        _ = combos.ilabels
        result = (
            combos
            .groupby('ilabels', observed=True, sort=False)
            .size()
            .reindex(self.ilabels, fill_value=0)
            .values
        )
        return result


    def contains_substring(self, substring: str) -> Series[bool]:
        return self.natural.str.contains(substring)
