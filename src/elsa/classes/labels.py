from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.classes.classes import Classes


class Labels(
    Resource,
    magic.Frame,
):
    outer: Classes

    def conjure(self) -> Self:
        """
        NxL matrix where N is number of samples and L is number of labels.
        """
        outer = self.outer.stacked
        names = 'ilabels ilabel'.split()

        ilabels = outer.ilabels.values
        ilabel = outer.ilabel.values
        arrays = ilabels, ilabel
        needles = pd.MultiIndex.from_arrays(arrays, names=names)
        assert not needles.duplicated().any()

        ilabel = self.elsa.labels.ilabel.unique()
        ilabels = outer.ilabels.unique()
        arrays = ilabels, ilabel
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

        # Create a DataFrame with a single row where all values are 0
        new_row = pd.DataFrame([0] * result.shape[1], index=result.columns).T
        new_row.index = [tuple()]

        # Append the new row to the result DataFrame
        result = pd.concat([result, new_row])
        result.index.name = 'ilabels'
        return result

    @magic.index
    def ilabels(self):
        ...

    def foresight(self):
        """
        classes.labels
        ilabel      0   1   2   3   4   5   6   7   ...  27  28  29  30  31  32  33  34
        ilabels                                     ...
        (0,)         1   0   0   0   0   0   0   0  ...   0   0   0   0   0   0   0   0
        (0, 3)       1   0   0   1   0   0   0   0  ...   0   0   0   0   0   0   0   0
        (0, 3, 10)   1   0   0   1   0   0   0   0  ...   0   0   0   0   0   0   0   0
        (0, 3, 11)   1   0   0   1   0   0   0   0  ...   0   0   0   0   0   0   0   0
        (0, 3, 14)   1   0   0   1   0   0   0   0  ...   0   0   0   0   0   0   0   0
        ...         ..  ..  ..  ..  ..  ..  ..  ..  ...  ..  ..  ..  ..  ..  ..  ..  ..
        (2, 5, 21)   0   0   1   0   0   1   0   0  ...   0   0   0   0   0   0   0   0
        (2, 5, 30)   0   0   1   0   0   1   0   0  ...   0   0   0   1   0   0   0   0
        (2, 5, 31)   0   0   1   0   0   1   0   0  ...   0   0   0   0   1   0   0   0
        (2, 5, 34)   0   0   1   0   0   1   0   0  ...   0   0   0   0   0   0   0   1
        (2, 34)      0   0   1   0   0   0   0   0  ...   0   0   0   0   0   0   0   1
        """
