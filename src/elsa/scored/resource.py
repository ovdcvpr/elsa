from __future__ import annotations

import magicpandas as magic

if False:
    import elsa.scored.scored

class Resource(magic.Magic):

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...


    @magic.cached.outer.property
    def scored(self) -> elsa.scored.scored.Scored:
        ...
