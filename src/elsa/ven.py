from __future__ import annotations

import os

import magicpandas as magic
import pandas as pd

from elsa.resource import Resource


class Method(
    magic.Magic
):
    @magic.column
    def total(self):
        """how many predictions left by a postprocess method"""

    @magic.column
    def tp(self):
        """how many true positives left by a postprocess method"""

    @magic.column
    def difference(self):
        """how many true positives are not left in the other postprocess method"""


class ScoreRange(magic.Magic):
    outer: DBA
    @magic.column
    def mean(self):
        """mean score range from the DBA predictions"""

    @magic.column
    def median(self):
        """median score range from the DBA predictions"""


class DBA(Method):
    @ScoreRange
    def score_range(self):
        ...


class Ven(
    Resource
):
    """
    Compare the outcomes of the postprocessing methods DBA and NMS
    on scored predictions. The table generated is essentially a
    "ven diagram"; you can determine the count of differing true
    positives and shared true positives between the two methods.
    """

    def __call__(
            self,
            *files,
            outdir='/tmp/ven',
            parallel=False,
            force=False,
    ):
        outdir = os.path.abspath(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        _files = [
            os.path.join(dir, file)
            for dir in files
            if os.path.isdir(dir)
            for file in os.listdir(dir)
            if file.endswith('.parquet')
        ]
        _files.extend(
            file
            for file in files
            if not os.path.isdir(file)
        )
        files = _files
        if parallel:
            scoreds = self.elsa.scored.from_outfiles(*files)
        else:
            elsa = self.outer

            def gen():
                for file in files:
                    yield elsa.scored(outfile=file)
                    del elsa.scored

            scoreds = gen()

        for file in files:
            name = os.path.basename(file)
            outpath = os.path.join(outdir, f'{name}.csv')
            if (
                    not force
                    and os.path.exists(outpath)
            ):
                continue
            scored = next(scoreds)
            nms = scored.nms()
            dba = scored.cdba()

            loc = dba.true_positive.values
            loc &= ~dba.ipred.isin(nms.ipred)
            dba_difference = loc.sum()

            loc = nms.true_positive.values
            loc &= ~nms.ipred.isin(dba.ipred)
            nms_difference = loc.sum()

            a = nms.true_positive.values
            b = dba.true_positive.values
            tp_intersection = nms.ipred[a].isin(dba.ipred[b])

            loc = dba.true_positive.values
            mean = dba.score_range.loc[loc].mean()
            median = dba.score_range.loc[loc].median()

            result = pd.DataFrame({
                'name': [name],
                'nms.total': [len(nms)],
                'nms.tp': [nms.true_positive.sum()],
                'nms.difference': [nms_difference],
                'tp_intersection': [tp_intersection.sum()],
                'dba.difference': [dba_difference],
                'dba.tp': [dba.true_positive.sum()],
                'dba.total': [len(dba)],
                'dba.score_range.mean': [mean],
                'dba.score_range.median': [median],
            })
            result.to_csv(outpath, index=False)

            del scored.cdba
            del scored.nms

        results = []
        for file in files:
            name = os.path.basename(file)
            outpath = os.path.join(outdir, f'{name}.csv')
            result = pd.read_csv(outpath)
            results.append(result)

        result = pd.concat(results)
        return result

    @magic.column
    def name(self):
        ...

    @Method
    def nms(self):
        ...

    @DBA
    def dba(self):
        ...

    @magic.column
    def tp_intersection(self):
        ...
