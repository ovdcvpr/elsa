from __future__ import annotations

from typing import *

from pandas._typing import Level, AxisInt
from pandas.core.generic import bool_t

import magicpandas as magic
from elsa.classes import has

if False:
    import elsa.classes.classes.Classes


class NUnique(
    has.ILabels,
    magic.Frame,
):
    def conjure(self) -> Self:
        return self.enchant(index=self.elsa.classes.index)

    @property
    def stacked(self):
        return self.classes.stacked

    @magic.column
    def states(self):
        loc = self.classes.stacked.cat == 'state'
        result = (
            self.classes
            .get_nunique_labels(loc)
            .reindex(self.ilabels, fill_value=0)
            .values
        )
        return result

