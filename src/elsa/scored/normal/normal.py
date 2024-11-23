from __future__ import annotations

from typing import *

import magicpandas as magic
from elsa.scored.cdba.cdba import CDBA
from elsa.scored.normal.matches import Matches


class Normal(CDBA):

    def __call__(
            self,
            iou: float = .80,
            anchored: bool = True,
            *args,
            **kwargs,
    ) -> Self:
        result = self.enchant(self.outer)
        result.threshold.iou = iou
        result.is_anchored = anchored
        return result

    @magic.cached.sticky.property
    def is_anchored(self) -> bool:
        return False

    @magic.column
    def true_positive(self) -> magic[bool]:
        result = (
            self.matches.true_positive
            .reindex(self.imatch, fill_value=False)
            .values
        )
        return result

    @Matches
    def matches(self):
        ...

