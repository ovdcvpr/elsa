from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.annotation.annotation import Annotation


class Labels(
    Resource,
    magic.Frame
):
    outer: Annotation

    @magic.index
    def iann(self) -> magic[int]:
        ...

    def conjure(self) -> Self:
        """
        NxL matrix where N is number of samples and L is number of labels.
        """
        outer = self.outer
        names = 'ibox ilabel'.split()

        ibox = outer.ibox.values
        ilabel = outer.ilabel.values
        arrays = ibox, ilabel
        needles = pd.MultiIndex.from_arrays(arrays, names=names)
        assert not needles.duplicated().any()

        ilabel = self.elsa.labels.ilabel.unique()
        ibox = outer.ibox.unique()
        arrays = ibox, ilabel
        haystack = pd.MultiIndex.from_product(arrays, names=names)

        data = (
            needles
            .isin(haystack)
            .astype(np.int8)
        )

        index = pd.Index(ilabel, name='ilabel')
        result = (
            Series(data, index=needles)
            .astype(np.int8)
            .unstack('ilabel', fill_value=0)
            .reindex(index, axis=1, fill_value=0)
        )

        return result

