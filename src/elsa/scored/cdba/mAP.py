from __future__ import annotations

import itertools
from functools import *
from typing import Self

import numpy as np
import pandas as pd

import magicpandas as magic
from elsa.scored import summary
from ultralytics.utils.metrics import ap_per_class

if False:
    from elsa.scored.cdba.cdba import CDBA


class Summary(
    summary.Summary
):
    outer: CDBA

    def conjure(self) -> Self:
        """
        Compute the AP for each level, condition, score, and IoU threshold.
        """
    @cached_property
    def scores(self):
        return np.linspace(80, .5, 96)
