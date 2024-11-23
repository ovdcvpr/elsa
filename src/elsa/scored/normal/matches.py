from __future__ import annotations

import magicpandas as magic
from elsa.scored.cdba import matches


class Matches(
    matches.Matches
):
    @magic.column
    def true_positive(self) -> magic[bool]:
        """
        Compute TP for matches which is a subset of Scored

        Compute TP, which requires:
            pred subclass ⊂ truth subclass
        """
        result = self.is_subcombo.values
        return result
