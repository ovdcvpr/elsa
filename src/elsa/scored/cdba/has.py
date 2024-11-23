from __future__ import annotations
import numpy as np

import magicpandas as magic
from elsa.scored.cdba.magic import Magic

if False:
    from elsa.truth.combos import Combos


class IGroup(
    magic.Frame,
    Magic
):
    @property
    def groups(self):
        return self.cdba.groups

    @magic.column
    def igroup(self) -> magic[int]:
        ...

    @magic.column
    def ngroup(self):
        result = (
            self.groups.ngroup
            .indexed_on(self.igroup.values)
            .values
        )
        return result

    @magic.column
    def ntrue(self):
        result = (
            self.groups.ntrue
            .indexed_on(self.igroup.values)
            .values
        )
        return result

    @magic.column
    def ngroup(self):
        result = (
            self.cdba.ngroup
            .indexed_on(self.igroup)
            .values
        )
        return result

    @magic.column
    def score_range(self):
        result = (
            self.groups.score_range
            .indexed_on(self.igroup)
            .values
        )
        return result

    @magic.column
    def score_max(self):
        result = (
            self.groups.score_max
            .indexed_on(self.igroup)
            .values
        )
        return result

    @magic.column
    def is_disjoint(self):
        result = (
            self.groups.is_disjoint
            .indexed_on(self.igroup, fill_value=False)
            .values
        )
        return result


class IPred(
    IGroup,
):

    @magic.column
    def iclass(self) -> magic[int]:
        result = (
            self.outer.iclass
            .loc[self.ipred.values]
            .values
        )
        return result

    @magic.column
    def ilabels(self):
        result = (
            self.cdba.ilabels
            .loc[self.ipred]
            .values
        )
        return result

    @magic.column
    def level(self):
        result = (
            self.cdba.level
            .indexed_on(self.ipred)
            .values
        )
        return result

    @magic.column
    def igroup(self):
        result = (
            self.cdba.igroup
            .indexed_on(self.ipred)
            .values
        )
        return result

    @magic.index
    def ipred(self):
        ...

    @property
    def includes(self):
        return self.cdba.includes

    @property
    def excludes(self):
        return self.cdba.excludes

    @magic.column
    def score(self):
        result = (
            self.cdba.score
            .loc[self.ipred]
            .values
        )
        return result

    @magic.column
    def ifile(self):
        result = (
            self.cdba.ifile
            .loc[self.ipred]
            .values
        )
        return result

    @magic.column
    def file(self):
        result = (
            self.cdba.file
            .loc[self.ipred]
            .values
        )
        return result


class IMatch(
    IPred
):
    @property
    def matches(self):
        return self.cdba.matches

    @magic.index
    def imatch(self) -> magic[int]:
        result = (
            self.matches
            .indexed_on(self.ipred)
            .values
        )
        return result

    @magic.cached.static.property
    def truth(self) -> Combos:
        """
        The subset of the ground truth combos DataFrame,
        aligned with the matches according to itruth.
        """
        result = (
            self.elsa.truth.combos
            .loc[self.itruth]
        )
        return result

    @magic.column
    def itruth(self):
        result = (
            self.matches.itruth
            .reindex(self.imatch, fill_value=-1)
            .values
        )
        return result

    @magic.column
    def true_positive(self) -> magic[bool]:
        result = (
            self.matches.true_positive
            .reindex(self.imatch, fill_value=False)
            .values
        )
        return result

    @magic.column
    def iou(self):
        result = (
            self.matches.iou
            .reindex(self.imatch, fill_value=1.)
            .values
        )
        return result

    @magic.column
    def ipred(self):
        result = (
            self.matches.ipred
            .loc[self.imatch]
            .values
        )
        return result

