from __future__ import annotations

from functools import cached_property
from typing import Self

import magicpandas as magic
from elsa.boxes import Boxes
from elsa.scored.cdba import has

if False:
    from elsa.scored.cdba.matches import Matches
    import elsa.scored.cdba.matches


class check(magic.column):
    """
    Column is being used as a check for disjoint combinations;
    -1: irrelevant
    0: ok
    1: disjoint

    """


class Disjoint(
    Boxes,
    has.IMatch,
    __call__=True,
):
    """
    Compute whether the matches are disjoint;
    These are no matches that are just false positives, but they are
    egregrious false positives, with which enough suggests the model
    has no understanding of the classification of the ground truth annotation,
    and any matches for that ground truth are simply due to chance.
    """
    outer: Matches

    def conjure(self) -> Self:
        result = self.enchant(index=self.matches.index)
        return result

    @magic.cached.sticky.property
    def checks(self):
        result = [
            key
            for key, value in self.__class__.__dict__.items()
            if isinstance(value, check)
        ]
        return result

    @magic.column
    def is_disjoint(self):
        result = (
            self[self.checks]
            .any(axis=1)
            .values
        )
        return result

    @check
    def condition(self):
        """
        Prediction is disjoint if the condition is not the same
        """
        matches = self.matches
        result = matches.truth.condition.values != matches.condition.values
        return result

    @check
    def state(self):
        """
        Prediction is disjoint if it contains too many states for
        the truth's condition

        if truth.condition == person:
            prediction.state.nunique() > 1
        if truth.condition == pair:
            prediction.state.nunique() > 2
        """
        matches = self.matches
        truth = matches.truth
        nstates = (
            self.elsa.classes.nunique_per_class.states
            .indexed_on(matches.ilabels)
            .values
        )

        a = truth.condition.values == 'person'
        a &= nstates > 1

        b = truth.condition.values == 'pair'
        b &= nstates > 2

        result = a | b
        return result

    @cached_property
    def threshold(self):
        return .3

    @magic.cached.outer.property
    def matches(self) -> elsa.scored.cdba.matches.Matches:
        ...
