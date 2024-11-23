from __future__ import annotations

import magicpandas as magic
from elsa.scored.ap.samples import MultiClass, MultiLabel
from elsa.scored.ap.summary import Summary

if False:
    from elsa.evaluation.evaluation import Evaluation


class AP(magic.Magic.Blank):
    outer: Evaluation

    @MultiLabel
    def multilabel(self):
        """Multilabel AP scores."""

    @MultiClass
    def multiclass(self):
        """Multiclass AP scores."""

    @Summary
    def summary(self):
        """
        DataFrame summarizing the multiclass and multilabel scores
        for the different subsets and averages.
        """
        magic.portal(Summary.__call__)
