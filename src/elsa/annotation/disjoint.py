from __future__ import annotations

from functools import *
from numpy import ndarray
from pandas import Series, DataFrame
from typing import *

import magicpandas as magic

if False:
    from elsa.root import Elsa
    from elsa.annotation.unique import Unique


class Check(magic.column):
    """Column is being used as a check for disjoint combinations"""


check = Union[Check, Series, ndarray]
globals()['check'] = Check


class Checks:
    def __set_name__(self, owner, name):
        self.__cache__: dict[type[Disjoint], set[str]] = {}
        self.__name__ = name

    def __get__(self, instance, owner: type[Disjoint]) -> set[str]:
        cache = self.__cache__
        if owner not in cache:
            disjoint = {
                key
                for base in owner.__bases__
                for cls in reversed(base.mro())
                if issubclass(cls, Disjoint)
                for key in cls.checks
            }
            disjoint.update(
                key
                for key, value in owner.__dict__.items()
                if isinstance(value, Check)
            )
            cache[owner] = disjoint
        return cache[owner]


class Disjoint(magic.Frame):
    outer: Unique
    owner: Unique
    checks = Checks()

    def conjure(self) -> Self:
        """Called when accessing Unique.disjoint to instantiate Invalid"""
        index = self.outer.index.unique()
        result = DataFrame(index=index)
        return result

    @magic.index
    def ilabels(self):
        ...

    @cached_property
    def unique(self):
        return self.outer

    @cached_property
    def elsa(self) -> Elsa:
        return self.outer.outer.outer

    @cached_property
    def label2ilabel(self) -> dict[str, int]:
        return self.elsa.synonyms.ilabel.to_dict()

    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> magic[bool]:
        """
        Determine if a combo includes a label
        Returns ndarrays to save time on redundant alignment
        """
        result = self.unique.includes(label, cat)
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> magic[bool]:
        """Determine if a combo excludes a label"""
        result = ~self.includes(label, cat)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """ For a given mask, count unique labels"""
        # self.outer.get_nunique_labels(loc)
        if isinstance(loc, Series):
            loc = loc.values
        result = (
            self.unique
            .get_nunique_labels(loc)
            .reindex(self.index, fill_value=0)
        )
        return result

    @check
    def c(self) -> check:
        """ multiple conditions """

        # disjoint where multiple conditions
        loc = self.unique.cat == 'condition'
        result = self.get_nunique_labels(loc) > 1

        return result

    @check
    def alone_crosswalk(self):
        """alone and crossing crosswalk"""
        a = self.includes('alone')
        a &= self.includes('crossing crosswalk')
        a &= self.includes('sitting')

        # disjoint only if condition, state, and activity
        b = self.includes(cat='condition')
        b &= self.includes(cat='state')
        b &= self.includes(cat='activity')

        result = a & b
        return result

    @check
    def sitting_and_standing(self):
        """sitting and standing"""
        a = self.includes('sitting')
        a &= self.includes('standing')
        a &= self.includes('person')
        return a

    @check
    def missing_standing_sitting(self):
        """missing standing or sitting"""
        a = self.excludes('sitting')
        a &= self.excludes('standing')
        b = self.includes('vendor')
        b |= self.includes('shopping')
        b |= self.includes('load/unload packages from car/truck')
        b |= self.includes('waiting in bus station')
        b |= self.includes('working/laptop')

        # disjoint only if condition, state, and activity
        c = self.includes(cat='condition')
        c &= self.includes(cat='state')
        c &= self.includes(cat='activity')

        return a & b & c

    @check
    def standing_sitting(self) -> check:
        """alone and multiple states"""

        # disjoint where alone and more than 1 state
        a = self.includes('alone')
        loc = self.unique.cat == 'state'
        a &= self.get_nunique_labels(loc) > 1

        return a

    @check
    def couple(self) -> check:
        """ couple and more than 2 states """
        # disjoint where couple and more than 2 states
        result = self.includes('couple')
        loc = self.unique.cat == 'state'
        result &= self.get_nunique_labels(loc) > 2
        return result

    @check
    def no_state(self):
        """no state in combo if anything other than condition"""
        a = self.includes(cat='activity')
        a |= self.includes(cat='others')
        a &= self.excludes(cat='state')
        a &= self.excludes('pet')
        return a

    @check
    def no_condition(self):
        """no condition in combo"""
        a = self.excludes(cat='condition')
        a &= self.excludes('pet')
        return a

    @property
    def all_checks(self) -> Self:
        for check in self.checks:
            getattr(self, check)
        return self
