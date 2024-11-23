from __future__ import annotations

from numpy import ndarray
from pandas import Series
from typing import *

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.classes.classes import Classes
    import elsa.classes.classes
    from elsa.classes.stacked import Stacked


class Check(magic.column):
    """Column is being used as a check for disjoint combinations"""


check = Union[Check, Series, ndarray]
globals()['check'] = Check


class Disjoint(
    Resource,
    magic.Frame,
):
    outer: Classes

    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        return self.outer.includes(label, cat)

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> Series[bool]:
        return self.outer.excludes(label, cat)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        return self.classes.get_nunique_labels(loc)

    @magic.cached.outer.property
    def classes(self) -> elsa.classes.classes.Classes:
        ...

    @property
    def checks(self) -> Self:
        columns = [
            key
            for key, value in self.__dict__.items()
            if isinstance(value, Check)
        ]
        result = self.loc[:, columns]
        return result

    @property
    def stacked(self) -> Stacked:
        return self.classes.stacked

    def conjure(self) -> Self:
        outer = self.outer
        result = self.enchant(index=outer.index)
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, Check):
                getattr(result, key)
        return result

    @check
    def c(self) -> check:
        """ multiple conditions """
        # disjoint where multiple conditions
        loc = self.stacked.cat == 'condition'
        result = self.get_nunique_labels(loc).values > 1
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
        loc = self.stacked.cat == 'state'
        a &= self.get_nunique_labels(loc) > 1

        return a

    @check
    def couple(self) -> check:
        """ couple and more than 2 states """
        # disjoint where couple and more than 2 states
        result = self.includes('couple')
        loc = self.stacked.cat == 'state'
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
