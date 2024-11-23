from __future__ import annotations

import itertools
from itertools import chain
from typing import Self

import numpy as np
from pandas import Series

import magicpandas as magic
from elsa.resource import Resource

if False:
    import elsa.annotation.upgrade
    import elsa.annotation.consumed
    from .unique import Unique


class Stacked(
    Resource,
    magic.Frame
):
    """A synonymous combination of labels for every unique combination of ilabel"""
    outer: Unique
    consumed: elsa.annotation.consumed.Consumed

    def conjure(self) -> Self:
        uniques = self.outer
        truth = uniques.outer
        elsa = truth.outer

        def cartesian(it_synonyms: Series):
            return list(itertools.product(*it_synonyms))

        try:
            synonyms = (
                # elsa.synonyms.prompts.synonyms
                elsa.synonyms.prompts
                .groupby('ilabel', sort=False)
                .isyn
                .apply(list)
                .loc[uniques.ilabel]
                .groupby(uniques.ilabels, sort=False, observed=True)
                .apply(cartesian)
            )
        except KeyError as e:
            loc = ~uniques.ilabel.isin(elsa.synonyms.prompts.ilabel)
            ilabel = uniques.ilabel.loc[loc].unique()
            loc = elsa.synonyms.ilabel.isin(ilabel)
            synonyms = elsa.synonyms.loc[loc].label
            msg = f'Some synonyms are likely missing from PROMPTS: {synonyms}'
            raise KeyError(msg) from e
        it = map(len, synonyms)
        first = np.fromiter(it, int, len(synonyms))
        it = map(len, chain.from_iterable(synonyms))
        count = first.sum()
        second = np.fromiter(it, int, count)
        ilabels = (
            synonyms.index
            .repeat(first)
            .repeat(second)
        )
        iprompts = np.arange(first.sum()).repeat(second)
        it = chain.from_iterable(chain.from_iterable(synonyms))
        count = second.sum()
        # label = np.fromiter(it, object, count)
        isyn = np.fromiter(it, int, count)
        # elsa.synonyms.prompts.rese
        label = (
            elsa.synonyms.prompts
            .reset_index()
            .set_index('isyn')
            .label
            .loc[isyn]
            .values
        )
        result = self.enchant({
            'ilabels': ilabels,
            'iprompt': iprompts,
            'label': label,
            'isyn': isyn,
        })
        _ = result.iorder, result.isyns
        result = (
            result
            .sort_values('ilabels iprompt iorder'.split())
            .set_index('ilabels iprompt'.split())
        )
        # assert not result.iprompt.duplicated().any()
        return result

    @magic.delayed
    def consumed(self) -> elsa.annotation.consumed.Consumed:
        """
        A subset of the stacked where:
            'consumer' labels have their natural label modified
            'consumed' labels are dropped
        """

    @magic.delayed
    def new_consumed(self) -> elsa.annotation.consumed.Consumed:
        """
        A subset of the stacked where:
            'consumer' labels have their natural label modified
            'consumed' labels are dropped
        """

    @magic.index
    def ilabels(self) -> magic[int]:
        """Identifier for each unique combination of ilabel"""

    @magic.index
    def iprompt(self) -> magic[int]:
        """Identifier for each prompt for each unique combination of ilabel"""

    @magic.column
    def iorder(self) -> magic[int]:
        result = (
            self.elsa.synonyms.iorder
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def label(self) -> magic[str]:
        ...

    @magic.column
    def natural(self):
        result = (
            self.elsa.synonyms.prompts.natural
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def ilabel(self):
        result = (
            self.elsa.synonyms.ilabel
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def cat(self):
        result = (
            self.synonyms.cat
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def is_vanilla(self) -> magic[bool]:
        result = (
            self.synonyms.is_vanilla
            .set_axis(self.synonyms.natural)
            .loc[self.natural]
            .values
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
            if isinstance(label, str):
                ilabel = self.synonyms.label2ilabel[label]
            elif isinstance(label, int):
                ilabel = label
            else:
                raise TypeError(f'label must be str or int, not {type(label)}')
            loc = self.ilabel == ilabel
        elif cat is not None:
            loc = self.cat == cat
        else:
            raise ValueError('label or cat must be provided')

        # noinspection PyTypeChecker
        return loc

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        return ~self.includes(label, cat)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ilabels and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        if isinstance(loc, Series):
            loc = loc.values
        result = Series(0, index=self.iprompt.unique())
        iprompt = self.loc[loc].iprompt.values
        update = (
            self.ilabel
            .loc[loc]
            .groupby(iprompt, sort=False, observed=True)
            .nunique()
        )
        result.update(update)
        return result

    def synonymous(
            self,
            label: str | list[str]
    ) -> Series[bool]:
        if isinstance(label, str):
            label = [label]
        ilabel = self.synonyms.ilabel.loc[label]
        loc = self.ilabel.isin(ilabel)
        return loc

    @magic.column
    def combo(self):
        result = (
            self.prompts.combo
            .set_axis(self.prompts.iprompt)
            .loc[self.iprompt]
            .values
        )
        return result

    @magic.column
    def cat_char(self) -> magic[str]:
        result = (
            Series({
                'condition': 'c',
                'state': 's',
                'activity': 'a',
                'others': 'o',
                None: ' ',
            })
            .loc[self.cat]
            .values
        )
        return result

    @magic.column
    def labelchar(self) -> magic[str]:
        _ = self.elsa.labels.char
        result = (
            self.elsa.labels
            .set_index('ilabel')
            .char
            .loc[self.ilabel]
            .values
        )
        return result

    @magic.column
    def prompt(self) -> magic[str]:
        """
        An intermediate string which will be concatenated to form the
        prompts.
        """
        return self.natural.values

    @magic.column
    def catchars(self) -> magic[str]:
        repeat = self.natural.str.len()
        result = (
            self.cat_char
            .str.repeat(repeat)
            .values
        )
        return result

    @magic.column
    def labelchars(self) -> magic[str]:
        repeat = self.natural.str.len()
        result = (
            self.labelchar
            .str.repeat(repeat)
            .values
        )
        return result

    @magic.column
    def isyn(self):
        ...

    @magic.column
    def isyns(self) -> magic[tuple[int]]:
        _ = self.ilabel
        result = (
            self
            .reset_index()
            .sort_values('ilabel')
            .groupby('iprompt', sort=False, observed=True)
            .isyn
            .apply(tuple)
            .loc[self.iprompt]
            .values
        )
        return result

    def contains_substring(self, substring: str) -> Series[bool]:
        return self.natural.str.contains(substring)
