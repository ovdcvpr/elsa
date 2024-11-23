from __future__ import annotations

import pandas as pd
from typing import *

import magicpandas as magic

if False:
    from magicpandas.drydoc.objects import Objects

"""
How we determine which ones to replace?

if (
    old.drydoc != drydocs.drydoc
    and old.magic == drydocs.magic
):
    drydoc has been modified
    update magic

if (

)

"""


class Old(magic.Magic.Blank):
    outer: Objects

    @magic.column
    def obj(self):
        objects = self.outer
        result = pd.Series(None, index=objects.index)
        return result

    @magic.column
    def drydoc(self):
        objects = self.outer
        result = pd.Series(None, index=objects.index)
        return result
