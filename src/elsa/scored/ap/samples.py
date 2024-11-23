from __future__ import annotations

import numpy as np

import magicpandas as magic
from elsa.scored.ap.average import Average

if False:
    from elsa.scored.ap.ap import AP
    import elsa.root
    import elsa.evaluation.evaluation


class Samples(magic.Magic):
    macro = Average()
    micro = Average()

    @magic.cached.static.property
    def y_true(self):
        """
        If the prediction iclass is a subcombo of the truth, the truth
        is set to the prediction iclass.
        """
        raise NotImplementedError

    @magic.cached.static.property
    def y_pred(self):
        """"""
        raise NotImplementedError

    @magic.cached.static.property
    def y_score(self):
        return self.eval.iou.values

    @magic.cached.outer.property
    def eval(self) -> elsa.evaluation.evaluation.Evaluation:
        ...

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...


class MultiClass(Samples):
    outer: AP

    @magic.cached.static.property
    def y_true(self):
        """
        If the prediction iclass is a subcombo of the truth, the truth
        is set to the prediction iclass.
        """
        truth = self.outer.multilabel.y_true
        pred = self.outer.multilabel.y_pred
        loc = ~np.any(pred & ~truth, axis=1)
        truth = (
            self.elsa.truth.combos.iclass
            .loc[self.eval.ibox]
            .values
        )
        pred = self.y_pred
        result = np.where(loc, pred, truth)
        return result

    @magic.cached.static.property
    def y_pred(self):
        result = (
            self.elsa.classes.iclass
            .loc[self.eval.ilabels]
            .values
        )
        return result

    @magic.cached.static.property
    def y_score(self) -> np.ndarray:
        result = self.eval.iou.values
        return result


class MultiLabel(Samples):
    outer: AP

    @magic.cached.static.property
    def y_true(self) -> np.ndarray:
        """
        2D array-like of shape (n_samples, n_labels). Contains the
        true binary labels indicating the presence of each label for
        each sample.
        """
        result = (
            self.eval.elsa.truth.labels
            .loc[self.eval.ibox]
            .values
        )
        return result

    @magic.cached.static.property
    def y_true(self):
        result = (
            self.eval.elsa.truth.labels
            .loc[self.eval.ibox]
            .values
        )
        return result

    @magic.cached.static.property
    def y_pred(self):
        result = (
            self.elsa.classes
            .loc[self.eval.ilabels]
            .values
        )
        return result

    @magic.cached.static.property
    def y_score(self) -> np.ndarray:
        result = self.eval.iou.values
        return result
