from __future__ import annotations
import numpy as np

import gc
import io
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from shapely import *

import magicpandas as magic
from elsa.classes import has
from elsa.scored.cdba.detection import Input

if False:
    from elsa.scored.cdba.detection import Input
    from elsa.scored.nms.nms import NMS
    from elsa.scored.cdba.cdba import CDBA


class APColumn(magic.column):
    outer: AP

    @magic.series
    def by_condition(self):
        ap = self.outer
        _ = ap['condition']
        result = (
            ap
            .reset_index()
            .groupby('condition', sort=False, observed=False)
            .ap
            .mean()
        )
        result['any_condition'] = ap.ap.mean()
        return result

    @magic.series
    def by_level(self):
        ap = self.outer
        _ = ap['level']
        result = (
            ap
            .reset_index()
            .groupby('level', sort=False, observed=False)
            .ap
            .mean()
        )
        result['any_level'] = ap.ap.mean()
        return result


class AP(
    has.ICls,
    magic.Frame
):
    outer: Summary

    @magic.index
    def iclass(self):
        ...

    @magic.index
    def iou(self):
        ...

    def conjure(self) -> Self:
        frame = self.outer
        groupby = 'threshold.iou pred.iclass'.split()

        # trapz is deprecated but we use old numpy
        try:
            # noinspection PyUnresolvedReferences
            trap = np.trapezoid
        except Exception:
            # noinspection PyUnresolvedReferences
            trap = np.trapz

        max = np.maximum.accumulate

        _ = frame['precision recall denominator'.split()]
        ious = self.outer.threshold.ious
        iclass = self.elsa.classes.iclass.values
        names = 'iclass iou'.split()
        arrays = [iclass, ious]
        index = pd.MultiIndex.from_product(arrays, names=names)

        def apply(frame: Self):
            # Precision must be monotonically decreasing
            precision = max(frame.precision.values[::-1])[::-1]
            # Compute AP using the trapezoidal rule
            recall = frame.recall.values
            ap = trap(precision, recall)
            return ap

        result = (
            frame
            .sort_values('recall', ascending=True)
            .groupby(groupby, sort=False, observed=False)
            .apply(apply)
            # turn series into frame
            .reset_index()
            # rename 0 to ap
            .rename(columns={
                0: 'ap',
                'pred.iclass': 'iclass',
                'threshold.iou': 'iou',
            })
            .sort_values('ap', ascending=False)
            .set_index(names)
            # necessary to include all the dropped groups as ap=0
            .reindex(index, fill_value=0)
            .pipe(self.enchant)
        )
        loc = result.iclass != 0
        result = result.loc[loc]
        return result

    @APColumn
    def ap(self):
        ...


class SIColumn(magic.column):
    @magic.series
    def by_condition(self):
        result = (
            self.outer
            .groupby('condition', sort=False, observed=False)
            .score
            .std()
        )
        return result

    @magic.series
    def by_level(self):
        result = (
            self.outer
            .groupby('level', sort=False, observed=False)
            .score
            .std()
        )
        return result


class SI(magic.Frame):
    outer: Summary

    def conjure(self) -> Self:
        # cdba = self.cdba
        groupby = 'threshold.iou ifile iclass'
        result = (
            cdba.groupby(
                'ifile iclass'.split(),
                sort=False,
                observed=True,
            )
            .score.std()
        )

    # @magic.cached.outer.property
    # def cdba(self) -> elsa.scored.cdba.cdba.CDBA:
    #     ...


class TruePositiveCounts(
    magic.Frame,
    has.ICls,
):
    def conjure(self) -> Self:
        ...


class Summary(
    Input
):
    outer: Input

    def conjure(self) -> Self:
        gc.collect()
        confs = self.threshold.scores
        ious = self.threshold.ious
        by = 'pred.iclass conf'.split()
        frame = self.outer.sort_values(by, ascending=False)
        frame['conf'] = frame['conf'].astype('float32').values
        frame['iou'] = frame['iou'].astype('float32').values
        frame['pred.iclass'] = frame['pred.iclass'].astype('int8').values
        frame['target.iclass'] = frame['target.iclass'].astype('int8').values
        frame['ipred'] = frame['ipred'].astype('int32').values

        # in recall accumulation, denominator is fixed
        nclass = (
            frame
            .groupby('pred.iclass', sort=False, observed=False)
            .size()
            .loc[frame.pred.iclass.values]
            .astype('int32')
            .values
        )
        frame['denominator'] = nclass
        loc = frame.pred.iclass.values != 0
        frame = frame.loc[loc]

        # broadcast each threshold conf to frame
        iloc = np.arange(len(frame)).repeat(len(confs))
        conf = np.tile(confs, len(frame))
        frame = (
            frame
            .iloc[iloc]
            .assign(**{'threshold.conf': conf, })
        )

        # broadcast each threshold iou to frame
        iloc = np.arange(len(frame)).repeat(len(ious))
        iou = np.tile(ious, len(frame))
        frame = (
            frame
            .iloc[iloc]
            .assign(**{'threshold.iou': iou, })
        )
        groupby = 'threshold.iou threshold.conf pred.iclass'.split()
        sort = groupby + ['conf']

        frame = frame.sort_values(sort, ascending=False)

        # threshold for each group
        loc = frame.iou.values >= frame.threshold.iou.values
        loc &= frame.conf.values >= frame.threshold.conf.values
        frame = frame.loc[loc]

        groups = frame.groupby(groupby, sort=False, observed=False)
        tp = (
            groups
            .tp
            .sum()
        )
        fp = (
            groups
            .size()
            .sub(tp)
        )
        den = (
            groups
            .denominator
            .first()
        )
        result = pd.DataFrame({
            'tp': tp.values,
            'fp': fp.values,
            'denominator': den.values,
        }, index=tp.index)
        return result

        size = (
            frame
            .reset_index()
            .groupby(groupby, sort=False, observed=False)
            .size()
            # .values
        )
        den = (
            frame
            .reset_index()
            .groupby(groupby, sort=False, observed=False)
            .denominator
            .first()
        )
        assert np.all(size <= den)

        # ilast = np.cumsum(size) - 1
        # ifirst = np.r_[0, ilast[:-1] + 1]

        # assert the indices are actually aligned

        CONF = frame.threshold.conf.values
        IOU = frame.threshold.iou.values
        ICLASS = frame.target.iclass.values
        DEN = frame.denominator.values
        for i, j in zip(ifirst, ilast):
            assert CONF[i] == CONF[j]
            # assert IOU[i] == IOU[j]
            assert ICLASS[i] == ICLASS[j]
            assert DEN[i] == DEN[j]
            assert DEN[i] >= (j - i + 1)

        return frame

    def conjure(self) -> Self:
        confs = self.threshold.scores
        ious = self.threshold.ious
        by = 'pred.iclass conf'.split()
        frame = self.outer.sort_values(by, ascending=False)
        frame['conf'] = frame['conf'].astype('float32').values
        frame['iou'] = frame['iou'].astype('float32').values
        frame['pred.iclass'] = frame['pred.iclass'].astype('int8').values
        frame['target.iclass'] = frame['target.iclass'].astype('int8').values
        frame['ipred'] = frame['ipred'].astype('int32').values

        # in recall accumulation, denominator is fixed
        nclass = (
            frame
            .groupby('pred.iclass', sort=False, observed=False)
            .size()
            .loc[frame.pred.iclass.values]
            .astype('int32')
            .values
        )

        # in recall, denominator is fn + tp
        #   we compute size while there are false negatives,
        #   and then exclude the fn from the rest of the process
        frame['denominator'] = nclass
        loc = frame.pred.iclass.values != 0
        FRAME = frame.loc[loc]

        iterations: list[pd.DataFrame] = []
        for iou in ious:
            loc_iou = FRAME.iou.values >= iou
            for conf in confs:
                loc_conf = FRAME.conf.values >= conf

                loc = loc_iou & loc_conf
                frame = FRAME.loc[loc]
                groups = frame.groupby('pred.iclass', sort=False, observed=False)
                tp = (
                    groups
                    .tp
                    .sum()
                )
                fp = (
                    groups
                    .size()
                    .sub(tp)
                )
                den = (
                    groups
                    .denominator
                    .first()
                )
                iteration = pd.DataFrame({
                    'tp': tp.values,
                    'fp': fp.values,
                    'denominator': den.values,
                    'threshold.iou': iou,
                    'threshold.conf': conf
                }, index=tp.index)
                iterations.append(iteration)

        index = 'threshold.iou threshold.conf pred.iclass'.split()
        result = (
            pd.concat(iterations)
            .reset_index()
            .set_index(index)
            .sort_values('tp', ascending=False)
            .pipe(self.enchant)
        )

        return result

    @magic.column
    def precision(self) -> magic[float]:
        result = (
                self.tp.values
                / (self.tp.values + self.fp.values)
        )
        return result

    @magic.column
    def recall(self) -> magic[float]:
        result = (
                self.tp.values
                / self.denominator.values
        )
        return result

    @recall.test
    def _test_recall(self):
        # assert self.recall.min() >= 0, f'{self.recall.min()}'
        # assert self.recall.max() <= 1, f'{self.recall.max()}'
        if not (
                (self.recall.min() >= 0)
                and (self.recall.max() <= 1)
        ):
            raise ValueError(f'{self.recall.min()} {self.recall.max()}')

    @precision.test
    def _test_precision(self):
        # assert self.precision.min() >= 0, f'{self.precision.min()}'
        # assert self.precision.max() <= 1, f'{self.precision.max()}'
        if not (
                (self.precision.min() >= 0)
                and (self.precision.max() <= 1)
        ):
            raise ValueError(f'{self.precision.min()} {self.precision.max()}')

    @AP
    @magic.portal(AP.conjure)
    def ap(self):
        ...

    @magic.frame
    def map(self):

        ap = self.ap
        _ = ap['level condition'.split()]
        map_level = (
            ap
            .groupby('level', sort=False, observed=False)
            .ap
            .mean()
        )
        map_condition = (
            ap
            .groupby('condition', sort=False, observed=False)
            .ap
            .mean()
        )
        result = (
            pd.concat([map_level, map_condition])
            .pipe(pd.DataFrame)
            .T
        )
        overall = (
            ap
            .ap
            .mean()
        )
        result.insert(0, 'overall', overall)
        index = pd.Index('map tp fp'.split(), name='metric')
        # todo: tp_counts, fp_counts seem wrong
        concat = result, self.tp_counts, self.fp_counts
        result = (
            pd.concat(concat)
            .set_axis(index, axis=0)
        )
        del result['']

        # nms: NMS
        # nms = self.outer.outer.outer
        # loc = self.tp.idxmax()
        # loc = self.pred.iclass == 1
        # self.loc[loc, 'tp denominator'.split()].sort_index()
        return result

    @has.ICls.Frame
    def si(self):
        """semantic inconsistency"""
        cdba = self.cdba
        loc = cdba.itruth.values != -1
        cdba = cdba.loc[loc]
        result = (
            cdba
            .groupby('iclass itruth ifile'.split(), observed=False)
            .score
            .std()
        )
        result = (
            result
            .loc[result.notna()]
            .to_frame('si')
        )
        return result

    @magic.frame
    def ss(self):
        """semantic similarity"""
        def apply(si: has.ICls.Frame) -> float:
            G = si.iclass.nunique()
            I = si.ifile.nunique()
            result = si.si.sum()
            # result /= G * I
            if result != 0:
                result /= G * I

            # with np.errstate(divide='raise', invalid='raise'):
            #     try:
            #         result /= G * I
            #     except FloatingPointError as e:
            #         raise ValueError("An invalid operation occurred in result /= G * I") from e
            # result /= len(si)
            result = 1 - result
            return result

        si = self.si
        _ = si['level condition'.split()]
        ss = apply(si)
        ss_level = (
            si
            .groupby('level', sort=False, observed=False)
            .apply(apply)
            .rename({
                'c': 'ss_c',
                'cs': 'ss_cs',
                'csa': 'ss_csa',
                'cso': 'ss_cso',
                'csao': 'ss_csao',
            })
        )
        ss_condition = (
            si
            .groupby('condition', sort=False, observed=False)
            .apply(apply)
            .rename({
                'person': 'ss_person',
                'pair': 'ss_pair',
                'people': 'ss_people',
            })
        )
        result = (
            pd.concat([ss_level, ss_condition])
            .pipe(pd.DataFrame)
            .T
        )
        result.insert(0, 'ss', ss)
        return result

    @magic.frame
    def tp_counts(self):
        scored = self.cdba
        level = (
            scored
            .true_positive
            .groupby(scored.level, observed=False)
            .sum()
        )
        condition = (
            scored
            .true_positive
            .groupby(scored.condition, observed=False)
            .sum()
        )
        overall = scored.true_positive.sum()
        index = pd.Index(['tp_counts'], name='metric')
        result = pd.DataFrame({
            'overall': overall,
            **level,
            **condition,
        }, index=index)
        del result['']
        return result

    @magic.frame
    def fp_counts(self):
        scored = self.cdba
        level = (
            scored.true_positive
            .__invert__()
            .groupby(scored.level, observed=False)
            .sum()
        )
        condition = (
            scored.true_positive
            .__invert__()
            .groupby(scored.condition, observed=False)
            .sum()
        )
        overall = scored.true_positive.__invert__().sum()
        index = pd.Index(['fp_counts'], name='metric')
        result = pd.DataFrame({
            'overall': overall,
            **level,
            **condition,
        }, index=index)
        del result['']
        return result

    @magic.frame
    def counts(self):
        scored = self.cdba
        level = (
            scored
            .groupby(scored.level, observed=False)
            .size()
        )
        condition = (
            scored
            .groupby(scored.condition, observed=False)
            .size()
        )
        overall = len(scored)
        index = pd.Index(['counts'], name='metric')
        result = pd.DataFrame({
            'overall': overall,
            **level,
            **condition,
        }, index=index)
        del result['']
        return result

    @magic.column
    def denominator(self) -> magic[int]:
        ...

    def plot_pr_curve(
            self,
            cls: int | str,
            iou: float,
            dark=True,
            show=True,
    ) -> Image.Image:
        method = self.cdba.name.upper()
        if isinstance(cls, str):
            classes = self.elsa.classes
            loc = classes.label.values == cls
            assert loc.any(), f"Class '{cls}' not found in classes."
            iclass = classes.iclass.loc[loc].iloc[0]
        else:
            iclass = int(cls)

        _ = self['target.label']
        loc = np.isclose(self.threshold.iou.values, iou)
        loc &= self.pred.iclass.values == iclass
        summary: Self = self.loc[loc]
        if not len(summary):
            # Return an empty plot if no precision or recall values are found
            plt.figure(figsize=(8, 6))
            if dark:
                plt.style.use('dark_background')
            else:
                plt.style.use('default')

            plt.title(
                f'Precision-Recall Curve using {method}\n'
                f'Class: {cls}; IoU: {iou}\n'
                f'No data available for this combination'
            )
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.grid(True)

            # Convert the plot to a PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            return img

        label = summary.target.label.values[0]

        precision = summary.precision.values
        recall = summary.recall.values
        i = np.argsort(recall)
        precision = precision[i]
        recall = recall[i]

        # Ensure the PR curve goes to zero at the maximum x value if it doesn't end at y=0
        recall = np.append(recall, recall[-1])
        precision = np.append(precision, 0.0)

        # Calculate AP
        ap = self.ap
        loc = ap.iclass == iclass
        loc &= np.isclose(ap.iou, iou)
        ap = ap.ap.loc[loc].values[0]

        # Monotonic version of precision for plotting
        monotonic_precision = np.maximum.accumulate(precision[::-1])[::-1]

        # Set the style to dark background if dark=True
        if dark:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        plt.figure(figsize=(8, 6))

        # Plot original PR curve in red
        plt.plot(recall, precision, color='red', lw=2)

        # Plot monotonic PR curve in blue
        plt.plot(recall, monotonic_precision, color='blue', lw=2)

        plt.title(
            f'Precision-Recall Curve using {method}\n'
            f'Class: {cls}; IoU: {iou:.2f}; AP: {ap:.6f}\n'
            f'Prompt: "{label}"'
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.legend()

        # Convert the plot to a PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        # plt.close()

        if show:
            plt.show()
        plt.close()
        # img.show()

        return img


def plot_pr_curve(
        self,
        cls: int | str,
        iou: float,
        dark=True,
        show=True,
) -> Image.Image:
    method = self.cdba.name.upper()
    if isinstance(cls, str):
        classes = self.elsa.classes
        loc = classes.label.values == cls
        assert loc.any(), f"Class '{cls}' not found in classes."
        iclass = classes.iclass.loc[loc].iloc[0]
    else:
        iclass = int(cls)

    _ = self['target.label']
    loc = np.isclose(self.threshold.iou.values, iou)
    loc &= self.pred.iclass.values == iclass
    summary: Self = self.loc[loc]
    if not len(summary):
        # Return an empty plot if no precision or recall values are found
        plt.figure(figsize=(8, 6))
        if dark:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        plt.title(
            f'Precision-Recall Curve using {method}\n'
            f'Class: {cls}; IoU: {iou}\n'
            f'No data available for this combination'
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)

        # Convert the plot to a PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    label = summary.target.label.values[0]

    precision = summary.precision.values
    recall = summary.recall.values
    i = np.argsort(recall)
    precision = precision[i]
    recall = recall[i]

    # Ensure the PR curve goes to zero at the maximum x value if it doesn't end at y=0
    recall = np.append(recall, recall[-1])
    precision = np.append(precision, 0.0)

    # Calculate AP
    ap = self.ap
    loc = ap.iclass == iclass
    loc &= np.isclose(ap.iou, iou)
    ap = ap.ap.loc[loc].values[0]

    # Monotonic version of precision for plotting
    monotonic_precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Set the style to dark background if dark=True
    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.figure(figsize=(8, 6))

    # Plot original PR curve in red
    plt.plot(recall, precision, color='red', lw=2, label='Original PR Curve')

    # Plot monotonic PR curve in blue
    plt.plot(recall, monotonic_precision, color='blue', lw=2, label='Monotonic PR Curve')

    plt.title(
        f'Precision-Recall Curve using {method}\n'
        f'Class: {cls}; IoU: {iou:.2f}; AP: {ap:.6f}\n'
        f'Prompt: "{label}"'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)

    # Only add the legend if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    # Convert the plot to a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    if show:
        plt.show()
    plt.close()

    return img
