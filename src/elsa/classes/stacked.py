from __future__ import annotations

from itertools import *
from typing import *

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.classes.has import ICls
from elsa.has import ILabel

if False:
    from elsa.classes.classes import Classes


class Stacked(
    ILabel,
    ICls,
    magic.Frame,
):
    outer: Classes

    def conjure(self) -> Self:
        outer = self.outer
        ilabels = outer.ilabels
        repeat = np.fromiter((
            map(len, ilabels)
        ), dtype=np.int64, count=len(ilabels))
        data = np.fromiter((
            chain.from_iterable(outer.ilabels)
        ), dtype=int, count=repeat.sum())
        index = ilabels.repeat(repeat)
        result = pd.DataFrame({
            'ilabel': data,
        }, index=index)
        return result

    @magic.index
    def ilabels(self):
        ...

    @magic.column
    def ilabel(self):
        ...

    @magic.column
    def cat_char(self):
        result = (
            self.elsa.labels.cat_char
            .indexed_on(self.ilabel)
            .values
        )
        return result

    @magic.column
    def cat(self):
        result = (
            self.elsa.labels.cat
            .indexed_on(self.ilabel)
            .values
        )
        return result


    def includes(
            self,
            label: str = None,
            cat: str = None,
    ) -> magic[bool]:
        """
        Determine if a combo includes a label
        Returns ndarrays to save time on redundant alignment
        """
        result = self.unique.includes(label, cat)
        return result

    def excludes(
            self,
            label: str = None,
            cat: str = None,
    ) -> magic[bool]:
        """Determine if a combo excludes a label"""
        result = ~self.includes(label, cat)
        return result


