from __future__ import annotations

import json
from pathlib import Path

import magicpandas as magic
from magicpandas.magic.magic import Magic
from magicpandas.magic.order import Order


class Locals(Magic):
    __order__ = Order.third


    @magic.cached.sticky.property
    def dict(self) -> dict[str]:
        ...

    def __init_nofunc__(self, obj, *args, **kwargs):
        if isinstance(obj, (str, Path)):
            with open(obj) as f:
                self.dict = json.load(f)
        else:
            self.dict = obj

