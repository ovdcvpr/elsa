from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.annotation.disjoint import Disjoint
from elsa.annotation.stacked import Stacked
from elsa.resource import Resource

if False:
    from .annotation import Annotation


class Unique(
    Resource,
    magic.Frame
):
    outer: Annotation
    stacked = Stacked()
    disjoint = Disjoint()

    def conjure(self) -> Self:
        """
        Called when accessing Annotation.unique to instantiate Unique
        """
        truth = self.outer
        names = 'ilabels ilabel'.split()
        _ = truth.ilabels, truth.iclass
        result = (
            truth
            .reset_index()
            .drop_duplicates(names)
            .set_index('ilabels')
        )
        return result

    consumed: Self

    def consumed(self):
        """The subset of Unique after consuming the labels."""
        names = 'ilabels ilabel'.split()
        arrays = self.ilabels, self.ilabel.values
        needles = pd.MultiIndex.from_arrays(arrays, names=names)

        consumed = self.stacked.consumed
        arrays = consumed.ilabels, consumed.ilabel.values
        haystack = pd.MultiIndex.from_arrays(arrays, names=names)

        loc = needles.isin(haystack)
        result = self.loc[loc]

        assert result.ilabels.isin(self.ilabels).all()
        return result

    @magic.column
    def ilabel(self) -> magic[int]:
        """an ilabel that belongs to the unique combination"""

    @magic.index
    def ilabels(self) -> magic[tuple[int]]:
        """all ilabels that belong to the unique combination"""

    @magic.column
    def iclass(self):
        ...

    @magic.column
    def cat(self):
        result = (
            self.synonyms
            .drop_duplicates('ilabel')
            .set_index('ilabel')
            .cat
            .loc[self.ilabel]
            .values
        )
        return result

    @magic.cached.sticky.property
    def tuples(self) -> magic[tuple[int]]:
        result = (
            self
            .reset_index()
            .sort_values('ilabel')
            .groupby('ilabels', sort=False)
            .ilabel
            .apply(tuple)
        )
        return result


    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        if label and cat:
            raise ValueError('label and cat cannot both be provided')

        if label is not None:
            ilabel = self.synonyms.ilabel.from_label(label)
            loc = self.ilabel == ilabel

        elif cat is not None:
            loc = self.cat == cat
            loc |= self.cat_char == cat
        else:
            raise ValueError('label or cat must be provided')
        result = (
            Series(loc)
            .groupby(self.index.names)
            .any()
        )
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        return ~self.includes(label, cat)

    def synonymous(self, label: str) -> Series[bool]:
        ilabel = self.synonyms.ilabel.loc[label]
        loc = self.ilabel == ilabel
        return loc

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        ann = self.outer
        result = Series(0, index=self.ilabels)
        ilabels = self.ilabels[loc]
        update = (
            ann.ilabel
            .loc[loc]
            .groupby(ilabels)
            .nunique()
        )
        result.update(update)
        result = result.set_axis(self.index)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        result = (
            self.ilabel
            .loc[loc]
            .groupby(level='ilabels')
            .nunique()
        )
        return result

    @magic.column
    def cardinal(self):
        result = (
            self.elsa.synonyms
            .drop_duplicates('ilabel')
            .reset_index()
            .set_index('ilabel')
            .label.loc[self.ilabel]
            .values
        )
        return result

    @magic.series
    def alone_appendix(self) -> magic[str]:
        """If only a singular subject is present, returns ' alone'"""

        # Later: we talked about only having "alone" with person! not with an individual.
        includes = self.includes
        excludes = self.excludes
        loc = (
                includes('alone')
                ^ includes('laborer')
                ^ includes('vendor')
                ^ includes('kid')
                ^ includes('teenager')
                ^ includes('elderly')
                ^ includes('baby')
        )
        loc &= excludes('couple')
        loc &= excludes('group')
        loc: Series[bool]
        data = np.where(loc, ' alone', '')
        result = Series(data, index=loc.index)
        return result
