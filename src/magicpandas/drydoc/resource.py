from __future__ import annotations

import magicpandas as magic

if False:
    import magicpandas.drydoc.magics


class Resource(magic.Magic.Blank):
    # class Resource(magic.Frame):
    @magic.cached.outer.property
    def magics(self) -> magicpandas.drydoc.magics.Objects:
        ...
