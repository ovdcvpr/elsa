from __future__ import annotations

from typing import Self

import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.scored.cdba.magic import Magic
from elsa.truth.combos import Combos

# todo: this should be in detection


class Truth(Magic):
    @magic.column
    def iclass(self) -> magic[int]:
        """
        0 where no class matched
        """
        truth = self.elsa.truth.combos
        cdba = self.cdba
        itruth = cdba.itruth
        loc = itruth != -1
        itruth = itruth[loc]
        iclass = truth.iclass.loc[itruth].values
        result = Series(0, index=self.cdba.index)
        result.loc[loc] = iclass
        return result

    @magic.column
    def ilabels(self) -> magic[tuple[int]]:
        """
        tuple() where no class matched
        """

    def unmatched(self):
        ...
