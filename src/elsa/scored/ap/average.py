from __future__ import annotations
import tqdm
import magicpandas as magic

from functools import *
from typing import *

import numpy as np
import pandas as pd
import time
import torch

from elsa.scored.resource import Resource

if False:
    from elsa.scored.ap.samples import Samples
    from elsa.scored.scored import Scored


class Average(
    magic.series,
    Resource,
    cached_property=True,
):
    outer: Samples

    @cached_property
    def iou_thresholds(self):
        return [0.75, 0.8, 0.85, 0.9]

    @cached_property
    def max_detection_thresholds(self):
        # is the second 1000 intended?
        return [100, 1000, 1000]

    def __call__(self, score_name: str):
        """
        Use torch to compute f1 score

        Here we iterate by class, and then by file.
        It is necessary to iterate by class so that we may "downgrade"
        the truth, if the prediction is a subclass is a truth. This is
        so that we may handle the case of:
            pred=person
            truth=person walking
        should be TP, but will be FP without intervention.
        """

        raise NotImplementedError('this is a dead end')
        from torchmetrics.detection import MeanAveragePrecision
        PREDICTIONS: Scored
        scored = PREDICTIONS = self.scored
        truth = TRUTH = self.elsa.truth.combos
        bounds = 'normw norms norme normn'.split()

        Y_PRED = torch.tensor(self.scored.iclass.values)
        getattr(self.scored, score_name)
        Y_SCORE = torch.tensor(self.scored[score_name].values)
        PBOX = torch.as_tensor(scored[bounds].values)
        tbox = torch.as_tensor(truth[bounds].values)
        preds = []
        target = []
        # YTRUE = self.elsa.truth.combos.subclasses.ytrue


        average = self.__name__
        if average == 'none':
            average = None
        # noinspection PyTypeChecker
        metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            iou_thresholds=self.iou_thresholds,
            max_detection_thresholds=self.max_detection_thresholds,
            average=average,
        )
        classes = (
            PREDICTIONS
            .reset_index()
            .groupby('iclass', observed=True, sort=False)
        )

        for iclass, pfirst in classes.indices.items():
            # ytrue = torch.tensor(YTRUE.loc[iclass].values)


            files = (
                scored
                .reset_index()
                .groupby('ifile', observed=True, sort=False)
            )
            for ifile, psecond in files.indices.items():
                boxes = PBOX[pfirst][psecond]
                scores = Y_SCORE[pfirst][psecond]
                labels = Y_PRED[pfirst][psecond]

                loc = truth.ifile.values == ifile


                p = dict(boxes=boxes, scores=scores, labels=labels)
                t = dict(boxes=tbox, labels=ytrue)

                preds.append(p)
                target.append(t)

        # for iclass in PREDICTIONS.iclass.unique():
        #     ...

        raise NotImplementedError
        # this is a dead end; we cannot iterate by class because
        #   it would duplicate truth within files
        """
        iterate by ifile 
        
        """




        # todo: truth also needs to be only for that particular file
        #

        metric.update(preds, target)
        compute = metric.compute()
        result = pd.Series({
            key: float(value)
            for key, value in compute.items()
            if not getattr(value, 'shape', True)
        })
        return result
