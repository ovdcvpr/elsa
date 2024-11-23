from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from functools import *
from pandas import Series
from typing import *

import magicpandas as magic
from elsa.synonyms.drop import DROP
from elsa.synonyms.order import ORDER
from elsa.synonyms.prompts import PROMPTS
from elsa.synonyms.cat import CAT
from elsa.synonyms.natural import NATURAL
from elsa.synonyms.incoco import INCOCO
from elsa.resource import Resource

if False:
    from elsa import Elsa


def syn(synonyms: list[str]) -> list[set[str]]:
    return [
        set(syn.split("; "))
        for syn in synonyms
    ]


SYNONYMS = syn([
    "alone; a person; person; individual; pedestrian; human",
    "at petrol/gas station; at gas station; at petrol station",
    "biking; cycling; riding bike; bicycling; riding bicycle",
    "construction-workers; laborers; builders; construction workers; construction worker; laborer; construction-worker",
    "couple/2people; two people; two pedestrians; two friends; couple; pair; two humans",
    "dining; eating; snacking",
    "GSV car interaction; car interaction; waving; beckoning",
    "group; people; crowd; gathering; many people",
    "on wheelchair; in wheelchair; using wheelchair; riding wheelchair",
    "pet interactions; with dog; with pet",
    # "phone interaction; using phone; on ",
    # note: on phone / using phone is ambiguous with "talking on phone" or "chatting on phone" and are not listed as synonyms here
    "phone interaction; playing with phone; looking at phone",
    "playing",
    'pushing stroller; pushing baby carriage; pushing pram; stroller; baby carriage pram; stroller; pushing stroller or shopping cart; pushing wheelchair or stroller',
    'pushing shopping cart; pushing cart; shopping cart',
    "running; jogging; sprinting",
    "shopping; buying; purchasing; browsing",
    "sitting; seated; sitting down; sitting on bench or chair; seated on bench or chair; sitting on bench; sitting on chair; sitting on chair or bench",
    "sport activities; sports; playing sports; athletics; athletic activities; exercising",
    "standing; standing up; standing upright",
    "street-vendors; vendors; street seller; merchant; vendor; street-vendor; street vendors; street vendor; street-merchant",
    "taking photo; taking picture; taking photograph; taking image; photographing",
    "talking; chatting; conversing; speaking; communicating; arguing; discussing; debating; conversing; dialoguing",
    "talking on phone; talking on cellphone; chatting on phone; chatting on cellphone; conversing on phone; conversing on cellphone; speaking on phone; speaking on cellphone; communicating on phone; communicating on cellphone",
    "waiting in bus station; waiting at bus station; waiting for bus; waiting for bus at bus station",
    "walking; strolling",
    "with bike; with bicycle",
    "with coffee or drinks; with coffee; with drink; with beverage",
    "baby/infant; baby; infant; newborn; toddler",
    "crossing crosswalk; crossing street; crossing road; crossing zebra crossing; crossing pedestrian crossing; crossing road",
    "duplicate; duplicated; duplication",
    "elderly; old; senior; aged; aged person; old person; senior person; elderly person",
    "kid; child; youngster",
    "with cane or walker; mobility aids; walking aids; walking stick; walking cane; crutches; wheelchair; walking frame; walking aid; mobility aid; with mobility aids",
    "model_hint",
    "multi-label; multiple labels; multiple label",
    "no people",
    "not sure/confusing; not sure; confusing; unsure; uncertain; ambiguous",
    "pet; service dog; guide dog",
    "public service/cleaning; public service; community service; cleaning",
    "riding carriage; riding horse carriage; riding horse cart; riding horse wagon; riding horse buggy; riding horse vehicle",
    "teenager; teen; adolescent; youth",
    "working/laptop; working on laptop; working with laptop; working on computer; working with computer; working on desktop; working with desktop",
    "no interaction",
    'police; law enforcement; police officer; cop',
    'load/unload packages from car/truck; loading packages; unloading packages; loading packages from car; unloading packages from car; loading packages from truck; unloading packages from truck; loading packages from vehicle; unloading packages from vehicle; load packages from car; unload packages from car; loading or unloading packages',
    'reading; reading book; reading newspaper; reading magazine; reading paper; reading document; reading text; reading article; reading journal; reading publication; reading publication',
    'with luggage; with suitcase',
    'waiting for food/drinks; waiting for food; waiting for drink; waiting for drinks; waiting for beverage; waiting for meal; waiting for snack; waiting for coffee; waiting for tea; waiting for food or drink; waiting for food or beverage; waiting for drink or beverage; waiting for meal or snack; waiting for coffee or tea',
    'taking cab/taxi; taking taxi; taking cab',
    'picnic; picnicking',
    'riding motorcycle; motorcycle riding; riding motorbike; motorbike riding; motorcycling; motorbiking',
    'hugging; embracing; cuddling; snuggling',
    "pushing wheelchair",
    'skating'
])

# we use synonyms to unify labels across datasets
#   however there are too many synonyms to be used when
#   generating prompts; we have a subset called prompts that
#   are just for prompt generation
# include the article for natural language generation

"""
get synonym.ilabel from natural or label
"""


class ILabel(magic.Column):
    outer: Synonyms

    def from_label(self, label: str | int) -> int:
        if isinstance(label, int):
            return label
        if not isinstance(label, str):
            raise TypeError(f'Expected str or int, got {type(label)}')
        outer = self.outer
        loc = outer.label.values == label
        loc |= outer.natural.values == label
        if not loc.any():
            raise ValueError(f'No label found for {label}')
        iloc = np.argmax(loc)
        result = outer.ilabel.iloc[iloc]
        return result


class Synonyms(
    Resource,
    magic.Frame
):
    """
    A DataFrame representing which labels are synonymous, and other
    catdata such as their category (condition, state, activity,
    other), natural representation (person -> a person), and whether
    these synonyms are used in the prompt generation.
    """
    outer: Elsa
    owner: Elsa
    prompts: Prompts
    drop_list: Self

    def conjure(self) -> Self:
        repeat = np.fromiter(map(len, SYNONYMS), int, len(SYNONYMS))
        igroup = np.arange(len(SYNONYMS)).repeat(repeat)
        count = repeat.sum()
        it = itertools.chain.from_iterable(SYNONYMS)
        syn = np.fromiter(it, dtype=object, count=count)
        index = (
            Series(syn, name="label")
            .str.casefold()
            .pipe(pd.Index)
        )
        assert not index.duplicated().any()
        result = self.enchant(dict(
            igroup=igroup,
        ), index=index)
        _ = result.isyn

        return result

    @magic.column
    def isyn(self) -> magic[int]:
        return np.arange(len(self))

    @magic.series
    def synonyms(self) -> Series[str]:
        """map ilabel to syns"""
        result = (
            self
            .reset_index()
            .set_index('ilabel')
            .label
        )
        return result

    def drop_list(self) -> Self:
        """A subset of synonyms which are meant to be dropped"""
        ilabel = self.ilabel.loc[DROP]
        loc = self.ilabel.isin(ilabel)
        result = self.loc[loc]
        return result

    def prompts(self) -> Prompts:
        """
        The subset of synoynms meant to be used for prompt generation;
        These aren't the actual prompts; they are the synonyms meant
        to be used in prompts.
        """

    @magic.index
    def label(self) -> Series[str]:
        """An undercase label that may be associated with a set of synonyms"""

    @magic.column
    def igroup(self):
        ...

    @magic.column
    def cat(self) -> Series[str]:
        """The category of the synonym set that the label belongs to"""

    @magic.column
    def cat_char(self) -> Series[str]:
        """First character of the category"""
        return self.cat.str[0]

    @magic.column
    def icat(self) -> Series[int]:
        """The unique index of the category of the synonym set that the label belongs to"""

    @cached_property
    def label2ilabel(self) -> dict[str, int]:
        return self.ilabel.to_dict()

    @magic.column
    def is_in_coco(self) -> magic[bool]:
        """Specific label is in COCO dataset"""
        result = self.label.isin(INCOCO)
        return result

    @magic.column
    def is_like_coco(self) -> magic[bool]:
        """Label is synonymous with a label in COCO dataset"""
        loc = self.is_in_coco
        ilabel = self.ilabel.loc[loc]
        loc = self.ilabel.isin(ilabel)
        return loc

    @cached_property
    def ilabel2syns(self) -> Series[list[str]]:
        result = (
            self.label
            .groupby(self.ilabel)
            .apply(list)
        )
        return result

    @cached_property
    def ilabel2ilabel(self) -> Series[str]:
        """
        Map each ilabel to an ilabel that is present in the truth;
        if the synonym is not present it is not included here.
        """

    @magic.column
    def iorder(self) -> Series[int]:
        """The natural order of the synonym in the prompt"""
        index = Series(ORDER.keys()).str.casefold()
        order = (
            Series(ORDER.values(), index=index, name='ilabel')
            .rename_axis('label')
        )
        loc = index.isin(self.label).values
        order = order.loc[loc]

        index = self.ilabel.loc[order.index]
        ilabel2order = (
            order
            .set_axis(index, axis=0)
            .groupby(level='ilabel')
            .first()
        )
        loc = ~self.label.isin(order.index)
        ilabel = self.ilabel.loc[loc]
        syn = self.label[loc]
        appendix = ilabel2order.loc[ilabel].set_axis(syn, axis=0)
        # categorical dtype where subject < prepositional < verb
        categories = 'subject prepositional verb'.split()
        dtype = pd.CategoricalDtype(categories, True)
        result = (
            pd.concat([order, appendix])
            .astype(dtype)
            .loc[self.label]
        )
        return result

    @magic.column
    def cat(self):
        cat = Series(CAT, name='cat')
        isyn = self.igroup.loc[cat.index]
        cat = (
            cat
            .set_axis(isyn, axis=0)
            .loc[self.igroup]
        )

        loc = self.igroup.isin(cat.index)
        if not loc.all():
            missing = self.igroup.values[~loc].tolist()
            formatted_missing = "\n".join(missing)
            raise ValueError(
                f'The following synonyms are missing from '
                f'the synonym-to-meta mapping:\n{formatted_missing}'
            )

        result = cat.values
        return result

    # @magic.column
    @ILabel
    def ilabel(self):
        labels = self.elsa.labels
        loc = self.label.isin(labels.label)
        syn = self.label[loc]
        isyn = self.igroup[loc]
        ilabel = labels.ilabel.loc[syn]
        ilabel = (
            ilabel
            .set_axis(isyn, axis=0)
            .reindex(self.igroup, fill_value=-1)
            .values
        )
        return ilabel

    @magic.column
    def natural(self) -> magic[str]:
        result = Series(self.label, index=self.label, name='natural')
        natural = Series(NATURAL, name='natural')
        loc = ~self.label.isin(natural.index)
        if loc.any():
            eg = self.label[loc].tolist()
            msg = f'The following labels are not in the natural metadata: {eg}'
            self.logger.info(msg)
        result.update(natural)
        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary label to represent synonyms"""
        result = (
            self.natural
            .groupby(self.ilabel.values)
            .first()
            .loc[self.ilabel]
            .values
        )
        return result


class Prompts(Synonyms):
    outer: Synonyms

    @magic.column
    def natural(self) -> Series[str]:
        """The natural language description of the synonym set"""
        result = (
            Series(NATURAL)
            .loc[self.label]
            .values
        )
        return result

    def conjure(self) -> Self:
        synonyms = self.outer
        repeat = np.fromiter(map(len, PROMPTS), int, len(PROMPTS))
        count = repeat.sum()
        it = itertools.chain.from_iterable(PROMPTS)
        syn = np.fromiter(it, dtype=object, count=count)
        syn = pd.Index(syn, name='label')
        loc = synonyms.label.isin(syn)
        result = synonyms.loc[loc]
        return result
