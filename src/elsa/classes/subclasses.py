from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic

if False:
    from elsa.classes.classes import Classes
    import elsa.classes.classes


class SubClasses(
    magic.Frame,
):
    """
    if cpred is a subclass of ctrue
        ytrue = cpred
    else
        ytrue = ctrue

    cpred   ctrue   ytrue

    """
    outer: Classes

    @magic.index
    def tclass(self) -> magic[int]:
        """iclass of Truth"""

    @magic.index
    def pclass(self) -> magic[int]:
        """iclass of Prediction"""

    @magic.column
    def ytrue(self) -> magic[int]:
        """ytrue for Truth to be used in evaluation"""

    @magic.column
    def tlabels(self) -> magic[tuple[int, ...]]:
        result = self._iclass2ilabels.loc[self.tclass].values
        return result

    @magic.column
    def plabels(self) -> magic[tuple[int, ...]]:
        result = self._iclass2ilabels.loc[self.pclass].values
        return result

    @magic.cached.sticky.property
    def _iclass2ilabels(self) -> Series[tuple[int, ...]]:
        result = (
            self.classes
            .iclass
            .reset_index()
            . set_index('iclass')
            .ilabels
        )
        return result

    def conjure(self) -> Self:
        # classes = self.elsa.classes.set_axis(self.classes.iclass)
        classes = self.classes.set_axis(self.classes.iclass)
        iterables = classes.iclass.values, classes.iclass.values
        # names = 'cpred ctrue'.split()
        pclass = self.pclass.name
        tclass = self.tclass.name

        names = pclass, tclass
        index = pd.MultiIndex.from_product(
            iterables=iterables,
            names=names
        )
        tclass = index.get_level_values(tclass)
        pclass = index.get_level_values(pclass)
        truth = classes.loc[tclass].values
        pred = classes.loc[pclass].values
        ytrue = tclass.values
        loc = ~np.any(pred & ~truth, axis=1)
        ytrue[loc] = pclass[loc]

        result = self.enchant({
            'ytrue': ytrue,
        }, index=index)
        _ = result['tlabels plabels'.split()]

        return result

    @magic.cached.outer.property
    def classes(self) -> elsa.classes.classes.Classes:
        ...

