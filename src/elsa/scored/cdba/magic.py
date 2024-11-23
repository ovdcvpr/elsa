from __future__ import annotations

import magicpandas as magic

if False:
    import elsa.scored.cdba.cdba


class Magic(magic.Magic):
    outer: elsa.scored.cdba.cdba.CDBA

    @magic.cached.outer.property
    def cdba(self) -> elsa.scored.cdba.cdba.CDBA:
        ...

    @property
    def matches(self):
        return self.cdba.matches

    @property
    def stacked(self):
        return self.cdba.stacked

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...
