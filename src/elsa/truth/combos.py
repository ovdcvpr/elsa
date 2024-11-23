from __future__ import annotations

import numpy as np

import elsa.combos.combos as combos
import magicpandas as magic
from elsa.classes.subclasses import SubClasses

if False:
    from .truth import Truth


class Combos(combos.Combos):
    outer: Truth

    @magic.cached.static.property
    def subclasses(self) -> SubClasses:
        subclasses = self.elsa.classes.subclasses
        name = subclasses.tclass.name
        result = subclasses.indexed_on(self.iclass, name=name)
        # assert that each group is still aligned with truth combos
        assert np.all(
            result.tlabels.loc[0] == self.ilabels.values
        )
        return result
