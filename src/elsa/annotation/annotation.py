from __future__ import annotations

import functools
from typing import Self

import numpy as np
import pandas as pd
from pandas import Series

import elsa.classes.has
import magicpandas as magic
from elsa.annotation.labels import Labels
from elsa.annotation.unique import Unique
from elsa.boxes import Boxes
from elsa.truth.combos import Combos


class Annotation(
    Boxes,
    elsa.classes.has.ICls
):
    unique = Unique()

    @Labels
    @magic.portal(Labels.conjure)
    def labels(self):
        """
        NxL matrix where N is number of samples and L is number of labels.
        """

    @Combos
    @magic.portal(Combos.conjure)
    def combos(self):
        """
        A DataFrame encapsulating the annotations aggregated by box.
        Three annotations representing 'person', 'walking', and 'on phone'
        will be aggregated into a single combo entry, with the label
        being 'person walking on phone'.
        """

    @magic.column
    def ilabels(self) -> Series[tuple[int]]:
        """
        An ordered tuple of the label IDs representing a given combo.
        For example, "person walking" would have ilabels=(0, 1), and
        "pedestrian strolling" would also have ilabels=(0, 1), as they are
        synonymous.
        """
        result = (
            self.combos.ilabels
            .subloc[self.ibox]
            .values
        )
        return result

    @magic.column
    def iclass(self) -> magic[int]:
        result = (
            self.elsa.classes.iclass
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.index
    def iann(self) -> magic[int]:
        """Each annotation has a unique identifier integer: iann"""
        return np.arange(len(self))

    @magic.column
    def ilabel(self) -> Series[int]:
        """
        Label ID of the label; synonymous labels will have the same
        ilabel value. For example, 'person' and 'individual' have the same
        ilabel value.
        """
        labels = self.elsa.labels
        if not labels.passed:
            raise AttributeError
        result = (
            Series(labels.ilabel, index=labels.label)
            .loc[self.label]
            .values
        )
        return result

    @magic.column.from_options(dtype='category')
    def label(self) -> magic[str]:
        """String label for each annotation assigned by the dataset"""
        labels = self.elsa.labels
        if not labels.passed:
            raise AttributeError
        result = (
            labels
            .reset_index()
            .set_index('ilabel')
            .label
            .loc[self.ilabel]
            .values
        )

        return result

    @magic.column.from_options(dtype='category')
    def cat(self) -> Series[str]:
        """Metaclass, or type of label, for each box"""
        result = self.synonyms.cat.loc[self.label].values
        return result

    @magic.column.from_options(dtype='category')
    def cat_char(self) -> Series[str]:
        """Metaclass, or type of label, for each box"""
        result = self.synonyms.cat_char.loc[self.label].values
        return result

    @magic.column
    def ibox(self) -> Series[int]:
        """
        Unique identifier for each combo box that the annotatin belongs
        to in aggregate
        """
        columns = 'normw norms norme normn'.split()
        needles = pd.MultiIndex.from_frame(self[columns])
        haystack = needles.unique()
        ibox = (
            Series(np.arange(len(haystack)), index=haystack)
            .loc[needles]
            .values
        )
        return ibox

    @magic.column.from_options(dtype='category')
    def combo(self) -> magic[str]:
        """
        String of combined labels for the combo box that the annotation
        belongs to in aggregate:

        label       combo
        person      person walking on phone
        walking     person walking on phone
        on phone    person walking on phone
        """
        result = self.combos.label.loc[self.ibox].values
        return result

    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        """
        Return a boolean mask for annotations that belong to a combo
        that include the label or category provided.
        """
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
            .groupby(self.ibox, sort=False)
            .any()
            .loc[self.ibox]
            .set_axis(self.index)
        )
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        """
        Return a boolean mask for annotations that belong to a combo
        that do not include the label or category provided.
        """
        result = ~self.includes(label, cat)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ilabels and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        stacked = self.stacked
        result = Series(0, index=self.iprompt)
        update = (
            self.ilabel
            .loc[loc]
            .groupby('iprompt', sort=False, observed=True)
            .nunique()
        )
        result.update(update)
        return result

    def contains_substring(self, substring: str) -> Series[bool]:
        return self.natural.str.contains(substring)

    def synonymous(self, label: str) -> Series[bool]:
        """True if the label is synonymous with the passed label"""
        ilabel = self.synonyms.ilabel.loc[label]
        loc = self.ilabel.values == ilabel
        return loc

    @magic.column
    def iorder(self) -> magic[int]:
        """
        Value assigned for ordering the labels in the process of
        generating natural prompts from the combinations.
        """
        result = (
            self.elsa.synonyms.iorder
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def natural(self) -> magic[str]:
        """
        Natural language representation of the label:
        'person' -> 'a person'
        'sports' -> 'doing sports'
        """
        result = (
            self.elsa.synonyms.natural
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def is_disjoint(self):
        """True where annotation is part of an invalid combo"""
        # return self.combos.is_disjoint.loc[self.ibox].values
        result = (
            self.combos.is_disjoint
            .loc[self.ibox]
            .values
        )
        return result

    @functools.cached_property
    def yolo(self) -> Self:
        """Return the annotations in YOLO format"""
        _ = self.normx, self.normy, self.normheight, self.normwidth, self.path
        columns = ['normx', 'normy', 'normheight', 'normwidth', 'path']
        return self[columns]

    @magic.column
    def num_labels(self) -> magic[int]:
        result = (
            self.ibox
            .groupby(self.ibox.values)
            .size()
            .loc[self.ibox.values]
        )
        return result
