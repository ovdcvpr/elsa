from __future__ import annotations

import numpy as np
import pandas as pd
from typing import *

import magicpandas as magic
from magicpandas.magic.drydoc.drydoc import DryDoc

if False:
    from magicpandas.drydoc.objects import Objects


class DryDocs(magic.Frame):
    outer: Objects

    def conjure(self) -> Self:
        drydoc = self.outer.drydoc.unique()
        index = pd.Index(drydoc, name='drydoc')
        return

    @magic.column
    def cls(self) -> Optional[DryDoc]:
        # noinspection PyTypeChecker
        classes: Iterable[magic.Magic] = self.outer.cls
        np.fromiter((
            cls.drydoc
            for cls in classes
        ), dtype=object, count=len(self.outer))

    @magic.column
    def name(self) -> magic[str]:
        ...

    @magic.column
    def old(self):
        ...



