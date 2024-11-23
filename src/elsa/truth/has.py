from __future__ import annotations

import magicpandas as magic
from elsa.resource import Resource

if False:
    import elsa.classes.classes
    from elsa.truth.truth import Truth


class IAnn(
    Resource
):
    @magic.cached.static.property
    def truth(self) -> Truth:
        return self.elsa.truth

    @magic.index
    def iann(self):
        ...

    @magic.column
    def ifile(self):
        result = (
            self.truth.ifile
            .loc[self.iann]
            .values
        )
        return result

    @magic.column
    def file(self):
        result = (
            self.truth.file
            .loc[self.iann]
            .values
        )
        return result
