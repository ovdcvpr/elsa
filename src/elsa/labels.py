from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series

import elsa.util as util
import magicpandas as magic
from elsa.resource import Resource

bing = Path(__file__, *'.. static bing label_ids.txt'.split()).resolve()
google = Path(__file__, *'.. static google label_ids.txt'.split()).resolve()
unified = Path(__file__, *'.. .. .. gt_data triple_inspected_May23rd merged label_id_dict_after_distr_thresholding.csv'.split()).resolve()


class Labels(
    Resource,
    magic.Frame
):
    """
    A DataFrame of the unique Labels used by the dataset, mapping
    their names to their IDs,
    """

    def conjure(self) -> Self:
        with self.configure:
            passed = self.passed
        result = self.from_inferred(passed)
        result.label = result.label.str.casefold()
        result = result.set_index('label')
        return result

    @magic.column
    def isyn(self) -> Series[int]:
        """
        The index of the set of synonyms that the label belongs to;
        this resolves ambiguities.
        """
        result = (
            self.synonyms.isyn
            .loc[self.label]
            .values
        )
        return result

    @magic.index
    def ilabel(self):
        """
        Label ID of the label; synonymous labels will have the same
        ilabel value. For example, 'person' and 'individual' have the same
        ilabel value.
        """

    @magic.index
    def label(self) -> Series[str]:
        """
        String label for each annotation assigned by the dataset.
        For an annotation this is the singular label e.g. 'person';
        For a combo this is the concatenated label e.g. 'person walking'
        """

    @magic.column
    def color(self):
        """Map a color to each label for visualization"""
        colors = util.colors
        assert len(self) <= len(colors)
        return colors[:len(self)]

    # todo: label, ilabel
    @classmethod
    def from_json(cls, path):
        with open(path) as file:
            data = json.load(file)
        cats: list[dict[str, str | int]] = data['categories']
        info = data['info']
        count = len(cats)
        ilabel = np.fromiter((
            cat['id']
            for cat in cats
        ), int, count=count)
        label = np.fromiter((
            cat['name'].casefold()
            for cat in cats
        ), object, count=count)

        index = pd.Index(ilabel, name='ilabel')
        result = cls(dict(
            label=label,
        ), index=index)
        result.year = info['year']
        result.version = info['version']
        result.contributor = info['contributor']
        return result

    @classmethod
    def from_txt(cls, path) -> Self:
        with open(path) as file:
            lines = file.readlines()
        count = len(lines)
        ilabel = np.arange(count)
        label = np.fromiter((
            line.strip()
            for line in lines
        ), object, count=count)
        index = pd.Index(ilabel, name='ilabel')
        result = cls(dict(
            label=label,
        ), index=index)
        return result

    @classmethod
    def from_csv(cls, path) -> Self:
        columns = dict(
            id='ilabel',
        )
        result = (
            pd.read_csv(path)
            .rename(columns=columns)
            .pipe(cls)
        )
        return result

    @classmethod
    def from_pickle(cls, path) -> Self:
        result = cls.from_pickle(path)
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        if isinstance(path, (Path, str)):
            path = Path(path)
            match path.suffix:
                case '.json':
                    result = cls.from_json(path)
                case '.csv':
                    result = cls.from_csv(path)
                case '.txt':
                    result = cls.from_txt(path)
                case _:
                    raise ValueError(f'Unsupported file type: {path.suffix}')
        elif path is None:
            result = cls()
        else:
            msg = f'Labels expected a Path or str, got {type(path)}'
            raise TypeError(msg)
        result.passed = path
        return result

    @magic.column
    def cat(self) -> Series[str]:
        result = (
            Series(self.label2cat)
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def icat(self) -> Series[int]:
        result = (
            Series(self.cat2icat)
            .loc[self.cat]
            .values
        )
        return result

    def decompose(self, labels: list[str]) -> list[list[list[str]]]:
        wrong2ambuigities = self.wrong2ambiguities
        result: list[list[list[str]]] = []
        LABEL: str
        for LABEL in labels:
            possibilities = [[]]
            label = str(LABEL)
            while label:
                for wrong, ambiguities in wrong2ambuigities.items():
                    if label.startswith(wrong):
                        label = (
                            label
                            .replace(wrong, '')
                            .strip()
                        )
                        possibilities = [
                            possibility + [amguity]
                            for possibility in possibilities.copy()
                            for amguity in ambiguities
                        ]
                        break
                else:
                    raise ValueError(f'Could not decompose {LABEL=}')
            result.append(possibilities)

        return result

    def get_ilabel(self, label: list[str] | Series[str]) -> ndarray[int]:
        result = (
            self
            .reset_index()
            .set_index('label')
            .ilabel
            .loc[label]
            .values
        )
        return result

    @cached_property
    def label2ilabel(self) -> Series[int]:
        result = Series(self.ilabel.values, index=self.label, name='ilabel')
        return result

    @magic.column
    def is_in_prompts(self):
        result = self.label.isin(self.synonyms.prompts.label)
        return result

    @magic.column
    def is_synonymous_in_prompts(self):
        result = self.isyn.isin(self.synonyms.prompts.ilabel)
        return result

    @magic.column.from_options(dtype='category')
    def cat(self) -> Series[str]:
        """The category, or type of label, for each box"""
        result = self.synonyms.cat.loc[self.label].values
        return result

    @magic.column
    def ord(self):
        result = self.ilabel.values + 65
        return result

    @magic.column
    def char(self):
        result = self.ord.map(chr).values
        return result

    @magic.series
    def char2label(self) -> Series[str]:
        index = self.char
        label = self.label
        result = Series(label, index=index)
        result[' '] = ''
        return result

    @magic.column
    def cat_char(self) -> magic[str]:
        result = (
            self.elsa.cat2char
            .loc[self.cat]
            .values
        )
        return result

    @magic.column
    def frequency(self) -> magic[int]:
        """
        The number of times each label was represented in the ground
        truth.
        """
        truth = self.truth
        _ = truth.ilabel
        result = (
            truth
            .groupby('ilabel', observed=True, sort=False)
            .size()
            .reindex(self.ilabel, fill_value=0)
            .values
        )
        return result

    @magic.column
    def natural(self):
        result = (
            self.synonyms.natural
            .loc[self.isyn]
            .values
        )
        return result


    def ilabels2condition(self, ilabels: pd.Series | pd.Index) -> ndarray:
        person = self.synonyms.ilabel.from_label('person')
        pair = self.synonyms.ilabel.from_label('pair')
        people = self.synonyms.ilabel.from_label('people')

        person = np.fromiter((
            person in tup
            for tup in ilabels
        ), bool, count=len(ilabels))
        pair = np.fromiter((
            pair in tup
            for tup in ilabels
        ), bool, count=len(ilabels))
        people = np.fromiter((
            people in tup
            for tup in ilabels
        ), bool, count=len(ilabels))

        none = ~person & ~pair & ~people
        assert (
                pair
                ^ person
                ^ people
                | none
        ).all()
        result = np.full_like(person, '', dtype=object)
        result = np.where(person, 'person', result)
        result = np.where(pair, 'pair', result)
        result = np.where(people, 'people', result)
        return result
