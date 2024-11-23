from __future__ import annotations
import gc

from functools import cached_property
from itertools import *
from typing import Self

import networkx
import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa import util
from elsa.boxes import Boxes
from elsa.scored.cdba.magic import Magic
from elsa.scored.cdba.has import IPred, IMatch
from elsa.classes.has import ICls
from elsa.scored.cdba.disjoint import Disjoint

if False:
    from elsa.scored.cdba.cdba import CDBA
    from elsa.truth.combos import Combos


class Matches(
    Boxes,
    IMatch,
    ICls,
    Magic,
    __call__=True,
):
    outer: CDBA

    @magic.column
    @magic.portal('alg.txt')
    def true_positive(self) -> magic[bool]:
        """
        Compute TP for matches which is a subset of Scored

        Compute TP, which requires:
            pred subclass ⊂ truth subclass
            pred group is not disjoint
        """
        self.igroup.isin(self.groups.igroup)
        result = self.is_subcombo.values & ~self.is_disjoint.values
        return result

    def conjure(self) -> Self:
        self.logger.info('Conjuring matches')
        scored: CDBA
        truth: Combos

        cdba = self.outer

        elsa = cdba.elsa
        truth = TRUTH = elsa.truth.combos
        _ = truth['area level ifile'.split()]
        _ = truth['area level ifile fw fs fe fn geometry file nfile'.split()]
        try:
            _ = cdba['geometry']
        except Exception:
            ...
        _ = cdba['area fw fs fe fn nfile'.split()]

        scored = LOGITS = cdba

        # only match prompts that are being assessed
        needles = cdba.prompt
        haystack = truth.prompts.natural
        loc = ~needles.isin(haystack.values)
        nunique = needles[loc].nunique()

        if nunique:
            eg = needles.unique()
            eg.isin(truth.prompts.natural)
            total = needles.nunique()
            msg = (
                f'{nunique} evaluation prompts out of {total} do not exist '
                f'in the ground truth prompts. This means that {loc.sum()} '
                f'evaluation boxes out of {len(needles)} will be dropped. '
            )
            self.logger.info(msg)
            scored = LOGITS = cdba.loc[~loc]

        # match truth and  logits
        ITRUTH, IPRED = util.sjoin(truth, scored)
        del TRUTH.geometry
        del LOGITS.geometry
        itruth, ipred = ITRUTH, IPRED
        TRUTH = TRUTH['xmin xmax ymin ymax'.split()]
        LOGITS = LOGITS['xmin xmax ymin ymax prompt'.split()]
        truth = TRUTH.iloc[itruth]
        scored = LOGITS.iloc[ipred]
        gc.collect()

        # select matches where IOU > threshold
        intersection = util.intersection(truth, scored)
        union = util.union(truth, scored)
        iou = intersection / union

        # see alg.txt 17-22
        magic.portal('alg.txt')
        # if iou < iou_thr, classify as miss
        loc = iou >= self.outer.threshold.iou
        itruth, ipred, iou = itruth[loc], ipred[loc], iou[loc]
        truth = TRUTH.iloc[itruth]
        scored = LOGITS.iloc[ipred]

        imatch = pd.Index(np.arange(len(scored)), name='imatch')
        scored = scored.copy()
        _ = scored.ipred
        result: Self = (
            scored
            .reset_index()
            .assign(
                iou=iou,
                itruth=truth.ibox.values,
            )
            .sort_values('itruth prompt'.split())
            .set_axis(imatch)
            .pipe(self.enchant)
        )
        columns = ['itruth']
        columns += [col for col in result.columns if col not in columns]
        result: Self = result[columns]

        # select the max IoU matches
        loc = (
            result
            .groupby('ipred')
            .iou
            .idxmax()
        )
        result = result.loc[loc]

        return result

    @cached_property
    def iou_anchor_threshold(self) -> float:
        return .9

    def __align__(self, owner: CDBA = None) -> Self:
        loc = self.ipred.isin(owner.ipred)
        result = self.loc[loc]
        return result


    @magic.column
    def iou(self) -> magic[float]:
        ...

    @magic.column
    def itruth(self):
        """index of the truth COMBOS (not annotations)"""

    @magic.index
    def ipred(self):
        """index of the scored"""

    @magic.cached.static.property
    def truth(self) -> Combos:
        """
        The subset of the ground truth combos DataFrame,
        aligned with the matches according to itruth.
        """
        return (
            self.elsa.truth.combos
            .loc[self.itruth]
        )

    @magic.column
    def truth_ilabels(self) -> magic[tuple[int]]:
        result = (
            self.elsa.truth.combos.ilabels
            .loc[self.itruth]
            .values
        )
        return result

    @magic.column
    def ianchor(self) -> magic[int]:
        """
        ipred of the logit which anchored the group to the truth box.
        """
        iloc = (
            Series(self.iou.values)
            .groupby(self.itruth.values)
            .idxmax()
            .loc[self.itruth.values]
            .values
        )
        result = self.ipred.values[iloc]
        return result

    @magic.cached.static.property
    def anchor(self) -> Self:
        result = (
            self.cdba
            .indexed_on(self.ianchor, name='ipred')
            .assign(
                itruth=self.itruth.values,
                imatch=self.imatch.values,
            )
        )
        return result

    @magic.column
    def igroup(self):
        if not self.cdba.is_anchored:
            self.logger.warn(
                f'Grouping without an anchor. This is not recommended. '
                f'With large predictions, this generates many comparisons between'
                f'predictions that match a single ground truth.'
            )
            """
            skip the anchoring, meaning  that we only group boxes that have mpre than 0.9 overlap with each other
            """
            matches = (
                self.reset_index()
                [[self.itruth.name]]
                .assign(iloc=np.arange(len(self)))
            )
            join = matches.merge(matches, on=self.itruth.name, suffixes=('_x', '_y'))
            ileft = join.iloc_x.values
            iright = join.iloc_y.values
            left = self.iloc[ileft]
            right = self.iloc[iright]
            intersection = util.intersection(left, right)
            union = util.union(left, right)
            iou = intersection / union
            loc = iou > self.iou_anchor_threshold
            ileft, iright = ileft[loc], iright[loc]

            # group based on iou > threshold
            g = networkx.Graph()
            edges = np.c_[ileft, iright]
            g.add_edges_from(edges)
            cc = list(networkx.connected_components(g))
            repeat = np.fromiter(map(len, cc), int, len(cc))
            igroup = np.arange(len(repeat)).repeat(repeat)
            count = repeat.sum()

            # each element is iloc of scored
            iloc_pred = np.fromiter(chain.from_iterable(cc), int, count)
            # each element is iloc of group
            iloc_group = np.arange(len(iloc_pred))[iloc_pred]
            igroup = igroup[iloc_group]

        else:
            _ = self.imatch
            _  = self['prompt itruth'.split()]
            intersection = util.intersection(self, self.anchor)
            np.all(self.anchor.itruth.values == self.itruth.values)
            intersection /= util.area(self)
            loc = intersection > self.iou_anchor_threshold
            g = networkx.Graph()

            # create pairs of ipred and itruth to group on
            ipred = self.ipred.values[loc]
            ianchor = self.ianchor.values[loc]
            edges = np.c_[ipred, ianchor]
            g.add_edges_from(edges)

            # create reflexive pairs of ipred to group on
            ipred = self.ipred.values[~loc]
            edges = np.c_[ipred, ipred]
            g.add_edges_from(edges)

            # use cc to generate groups; map ipred to groups
            cc: list[set[int]] = list(networkx.connected_components(g))
            repeat = np.fromiter(map(len, cc), int, len(cc))
            igroup = np.arange(len(repeat)).repeat(repeat)
            count = repeat.sum()
            ipred = np.fromiter(chain.from_iterable(cc), int, count)
            index = pd.Index(ipred, name=self.ipred.name)
            igroup = (
                Series(igroup, index=index)
                .loc[self.ipred.values]
                .values
            )

        return igroup

    # @igroup.test
    # def _test_igroup_in_group(self):
    #     assert self.igroup.isin(self.cdba.groups.igroup).all()

    @magic.column
    def is_grouped(self) -> magic[bool]:
        result = (
            self
            .groupby(self.igroup.values)
            .size()
            .gt(1)
            .astype(bool)
            .loc[self.igroup.values]
            .values
        )
        return result

    @magic.test
    def _test_itruth(self):
        assert np.all(self.itruth.values == self.truth.ibox.values)

    @igroup.test
    def _test_igroup_same_itruth(self):
        # assert each group only has 1 unique truth
        loc = (
            self
            .groupby(self.igroup.values)
            .itruth
            .nunique()
            .eq(1)
        )
        if not loc.all():
            count = (~loc).sum()
            eg = loc.index[~loc].to_list()
            msg = (
                f'{count} groups have more than one unique truth, '
                f'e.g. {eg}'
            )
            raise AssertionError(msg)

    @magic.index
    @magic.portal(conjure)
    def imatch(self):
        ...

    @Disjoint
    @magic.portal(Disjoint.conjure)
    def disjoint(self):
        """
        Compute whether the matches are disjoint;
        These are no matches that are just false positives, but they are
        egregrious false positives, with which enough suggests the model
        has no understanding of the classification of the ground truth annotation,
        and any matches for that ground truth are simply due to chance.
        """

    @magic.column
    def is_subcombo(self) -> magic[bool]:
        arrays = self.ilabels, self.truth.ilabels
        loc = pd.MultiIndex.from_arrays(arrays)
        result = self.elsa.classes.is_subcombo.loc[loc].values
        return result

    @magic.column
    def score(self):
        result = (
            self.cdba
            .groupby('ipred')
            .score
            .first()
            .loc[self.ipred.values]
            .values
        )
        return result
