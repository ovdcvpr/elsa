from __future__ import annotations

from typing import *

from pandas import Series

import magicpandas as magic
from elsa.resource import Resource

if False:
    import elsa.classes.classes


class ILabels(
    Resource
):
    third: magic.Frame

    @property
    def classes(self) -> elsa.classes.classes.Classes:
        return self.elsa.classes

    @magic.index
    def ilabels(self):
        """
        Tuple of sorted ilabels to represent each unique combo of labels.

        labels             ilabels
        person             0
        person walking     0, 4
        """
        result = (
            self.classes
            .reset_index()
            .ilabels
            .loc[self.iclass]
            .values
        )
        return result

    @magic.index
    def iclass(self):
        """
        Identifier for each unique class (combo)

        labels              iclass
        person              0
        person walking      1
        group talking       2
        """
        third = self.third
        assert (
                'ilabels' in third.columns
                or 'ilabels' == third.index.name
                or 'ilabels' in third.index.names
        )
        result = (
            self.classes.iclass
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def condition(self):
        """
        The mutually exclusive value for the condition category within
        a combo:
            person, pair, or people
        """
        result = (
            self.classes.condition
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column.from_options(dtype='category')
    def level(self):
        """
        The level of the combo, or the characters, e.g. cs, csa, csaoa

        c: condition
        cs: condition, state
        csa: condition, state, activity
        cso: condition, state, others
        csao: condition, state, activity, others
        """
        result = (
            self.classes.level
            .reindex(self.ilabels, fill_value='')
            .values
        )
        return result

    @magic.column
    def label(self):
        """String label for each annotation assigned by the dataset"""
        result = (
            self.classes.label
            .reindex(self.ilabels.values)
            .values
        )
        return result

    @magic.cached.static.property
    def c(self) -> Self | magic.Frame:
        """Subset of the DataFrame where level == c"""
        loc = self.level == 'c'
        return self.third.loc[loc]

    @magic.cached.static.property
    def cs(self) -> Self | magic.Frame:
        """Subset of the DataFrame where level == cs"""
        loc = self.level == 'cs'
        return self.third.loc[loc]

    @magic.cached.static.property
    def csa(self) -> Self | magic.Frame:
        """Subset of the DataFrame where level == csa"""
        loc = self.level == 'csa'
        return self.third.loc[loc]

    @magic.cached.static.property
    def cso(self) -> Self | magic.Frame:
        """Subset of the DataFrame where level == cso"""
        loc = self.level == 'cso'
        return self.third.loc[loc]

    @magic.cached.static.property
    def csao(self) -> Self | magic.Frame:
        """Subset of the DataFrame where level == csao"""
        loc = self.level == 'csao'
        return self.third.loc[loc]

    @magic.cached.static.property
    def person(self) -> Self | magic.Frame:
        """Subset of the DataFrame where condition == person"""
        loc = self.condition == 'person'
        return self.third.loc[loc]

    @magic.cached.static.property
    def pair(self) -> Self | magic.Frame:
        """Subset of the DataFrame where condition == pair"""
        loc = self.condition == 'pair'
        return self.third.loc[loc]

    @magic.cached.static.property
    def people(self) -> Self | magic.Frame:
        """Subset of the DataFrame where condition == people"""
        loc = self.condition == 'people'
        return self.third.loc[loc]

    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        """
        Boolean mask for the classes which contain the label or category
        """
        result = (
            self.elsa.classes
            .includes(
                label=label,
                cat=cat,
            )
            .loc[self.ilabels]
            .values
        )
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        """
        Boolean mask for the classes which do not contain the label or category
        """
        return ~self.includes(label, cat)

    Frame: type[Frame]


class Frame(
    ILabels,
    magic.Frame
):
    ...


ILabels.Frame = Frame


class ICls(ILabels):
    ...
