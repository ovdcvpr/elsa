from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
import os
from time import time

from functools import *
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pandas import Series
from sklearn.cluster import DBSCAN
from torchvision.ops import nms

import magicpandas as magic
from elsa import boxes
from elsa.annotation.labels import Labels
from elsa.classes.has import ICls
from elsa.prediction.prediction import Prediction
from elsa.scored.scores import Scores
from elsa.scored.view import View

E = RecursionError, AttributeError

if False:
    from elsa.root import Elsa
    import elsa.scored.cdba.cdba
    import elsa.scored.nms.nms
    import elsa.scored.normal.normal


class Scored(
    boxes.Boxes,
    ICls,
):
    outer: Elsa
    cdba: elsa.scored.cdba.cdba.CDBA
    nms: elsa.scored.nms.nms.NMS
    normal: elsa.scored.normal.normal.Normal

    if False:
        # sorry for this mess, will look how to better handle this situation
        # just take the portal
        magic.portal(elsa.scored.nms.nms.NMS.__call__)
        nms = elsa.scored.nms.nms.NMS()
        magic.portal(elsa.scored.cdba.cdba.CDBA.__call__)
        cdba = elsa.scored.cdba.cdba.CDBA()
        magic.portal(elsa.scored.normal.normal.Normal.__call__)
        normal = elsa.scored.normal.normal.Normal()

    @magic.delayed
    def cdba(self) -> elsa.scored.cdba.cdba.CDBA:
        magic.portal(elsa.scored.cdba.cdba.CDBA.__call__)
        """Applies CDBA to the predictions"""

    @magic.delayed
    def nms(self) -> elsa.scored.nms.nms.NMS:
        magic.portal(elsa.scored.nms.nms.NMS.__call__)
        """Applies NMS to the predictions"""

    @magic.delayed
    def normal(self) -> elsa.scored.normal.normal.Normal:
        magic.portal(elsa.scored.normal.normal.Normal.__call__)
        """Applies nothing to the predictions"""

    @Scores
    def scores(self):
        ...

    # @AP
    # def ap(self):
    #     magic.portal(self.ap.summary.__call__)

    # todo: can we have multiple constructors? Here we have __call__,
    #   can we have Scored.from_directory, scored.from_blahblahblah,
    #   etc?
    Score = Union[
        Literal[
            'whole.nlse',
            'whole.argmax',
            'whole.soft_token',
            'whole.sigmoid_nlse',
            'selected.nlse',
            'selected.argmax',
            'selected.soft_token',
            'selected.sigmoid_nlse',
        ],
        str,
    ]

    # noinspection PyMethodOverriding
    def __call__(
            self,
            outfile: Path | str = None,
            indir: Path | str = None,
            infile: Path | str = None,
            score: Score = 'selected.nlse',
            compare: Union[str, list[str]] = (
                    'selected.nlse',
                    'whole.argmax',
            ),
            score_threshold: float = .3,
            force: bool = False,
            force_prompts: bool = True,
            force_files: bool = True,
            multiprocessing: bool = True,
            check_prompts=True,
    ) -> Scored:
        """
        outfile:
            Path to the output parquet file.
            If it exists, it will be loaded.
            If None, it will not be saved.
        indir:
            Path to the directiory of prediction parquets.
            If None, it will be loaded from outfile.
        infile:
            Path to a scored parquet file. The score name must be
            the same across the two files.
        score:
            Main score to use for thresholding.
        compare:
            Additional scores to include for comparison.
        score_threshold:
            Threshold for the score.
        force:
            Concatenate the predictions and save to file once more.
        strict:
            Restrict predictions to only ones with prompts currently
            in ELSA's set of prompts.
        multiprocessing:
            Use multiprocessing for loading predictions.
            False to help debug.
        check:
            Checks that the prediction labels also have the same
            identifiers as defined by the ELSA instance.
        """
        if isinstance(compare, str):
            compare = {compare}
        elif compare is None:
            compare = {score}
        else:
            compare = set(compare)
        compare.add(score)
        compare = list(compare)
        save = False

        elsa = self.outer
        if (
                outfile is None
                and indir is None
                and infile is None
        ):
            msg = f'Either infile, indir, or outfile must be specified'
            raise ValueError(msg)
        if outfile is not None:
            outfile = Path(outfile)
        if indir is not None:
            indir = Path(indir)

        if (
                force
                or outfile is None
                or not outfile.exists()
        ) and (
                infile is not None
        ):
            save = True

            result: Self = (
                pd
                .read_parquet(infile)
                .pipe(self.enchant)
            )
            loc = result.score >= score_threshold
            result = result.loc[loc].copy()

        elif (
                force
                or outfile is None
                or not outfile.exists()
        ) and (
                indir is not None
        ):
            save = True
            if indir is None:
                raise ValueError('directory must be specified')
            inpaths = [
                path
                for path in
                Path(indir)
                .expanduser()
                .resolve()
                .rglob('*.parquet')
                if pq.read_table(path, columns=[]).num_rows > 0
            ]
            if not inpaths:
                raise ValueError(f'no parquet files found in {indir}')
            # todo: this needs to be parallelized
            result = Prediction.from_inpaths(
                inpaths=inpaths,
                score=score,
                threshold=score_threshold,
                multiprocessing=multiprocessing,
                compare=compare,
            )
            result = self.enchant(result)
            result.elsa = elsa
        else:
            result = (
                pd
                .read_parquet(outfile)
                .pipe(self.enchant)
            )
            if not len(result):
                self.logger.warn(f'{outfile} is empty!')
            if 'ilabels' in result:
                result['ilabels'] = result.ilabels.map(tuple)

        result = self.enchant(result)
        result.elsa = elsa
        result.columns = result.columns.get_level_values(0)
        result.columns.name = None
        result = result.reset_index(drop=True)
        result.prompt = result.prompt.astype(dtype='category')
        result.file = result.file.astype(dtype='category')
        try:
            result.path = result.path.astype(dtype='category')
        except Exception:
            ...
        try:
            result.logit_file = result.logit_file.astype(dtype='category')
        except Exception:
            ...
        # todo: why does this not set file prompt as the index but rather set a column 'index' as (file, prompt) tuples?
        ipred = np.arange(len(result))
        result['ipred'] = ipred
        result = result.set_index('ipred')
        result.threshold = score_threshold

        # if nms:
        #     msg = f'Applying prompt-specific NMS'
        #     self.logger.info(msg)
        #     before = len(result)
        #     t = time()
        #     result = result.prompt_nms
        #     after = len(result)
        #     msg = (
        #         f'Applied prompt-specific NMS, {before} -> {after}, '
        #         f'took {time() - t:.2f} seconds'
        #     )
        #     self.logger.info(msg)


        if (
                save
                and outfile is not None
        ):
            msg = f'Writing to {outfile}, this will take some time.'
            self.logger.info(msg)
            result.to_parquet(outfile)
            self.logger.info(f'Wrote to {outfile}')

        if check_prompts:
            # check that the prompts are in accord with the elsa prompts
            loc = result.prompt.isin(elsa.prompts.natural).values
            result: Self = result.loc[loc].copy()
            iclass = (
                elsa.prompts.iclass
                .indexed_on(result.prompt, name='natural')
                .values
            )
            result.iclass = iclass
            del result.ilabels

        if force_files:
            loc = result.ifile.isin(elsa.files.ifile).values
            result = result.loc[loc].copy()

        if force_prompts:
            # forces the prompts to reflect those in ELSA
            loc = ~result.prompt.drop_duplicates().isin(elsa.prompts.natural)
            if loc.sum():
                msg = (
                    f"{loc.sum()} prediction prompts out of {len(loc)} total "
                    "are not in ELSA's prompts"
                )
                self.logger.warning(msg)
            loc = result.prompt.isin(elsa.prompts.natural.values)
            result = result.loc[loc].copy()

        if not len(result):
            self.logger.warn(f'{self.trace} is empty')

        loc = result.width.values <= 0
        loc |= result.height.values <= 0
        if loc.any():
            n = loc.sum()
            msg = f'{n} predictions out of {len(loc)} have width or height <= 0'
            self.logger.warning(msg)
            result = result.loc[~loc].copy()

        return result

    def from_outfiles(
            self,
            *files,
    ) -> Iterator[Self]:
        """
        Generates scored instances from outfiles iteratively.
        """
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
        futures = []
        with ThreadPoolExecutor() as threads:
            for file in _files:
                future = threads.submit(pd.read_parquet, file)
                futures.append(future)
            for future in futures:
                scored = future.result().pipe(self.enchant)
                yield scored





    @magic.column
    def epsilon(self):
        result = (
                self.image_width.values
                * self.image_height.values
                / 2
                * .05
        )
        return result

    def prompt_nms(self) -> 'Self':
        """Apply NMS for each class for each file"""
        _ = self['xmin ymin xmax ymax epsilon score'.split()]
        loc = []

        # Group by file and class
        for _, group in self.groupby(['ifile', 'iclass']):
            BBOX = group[['xmin', 'ymin', 'xmax', 'ymax']].values
            SCORE = torch.tensor(group['score'].values)
            EPSILON = group['epsilon'].iloc[0]

            # DBSCAN clustering on bounding boxes
            dbscan = DBSCAN(eps=EPSILON, min_samples=1)
            clusters = dbscan.fit_predict(BBOX)

            # Apply NMS for each cluster
            for cluster_id in set(clusters):
                if cluster_id == -1:
                    continue  # Skip noise points

                # Get indices of the current cluster
                cluster_indices = (clusters == cluster_id)
                bbox = torch.tensor(BBOX[cluster_indices], dtype=torch.float32)
                score = SCORE[cluster_indices]
                nms_indices = nms(bbox, score, iou_threshold=0.8)

                # Map back to original DataFrame indices
                original_indices = (
                    group.index
                    [cluster_indices]
                    [nms_indices]
                    .tolist()
                )
                loc.extend(original_indices)

        # Return a DataFrame with only the kept indices
        loc = np.array(loc)
        result = self.loc[loc]
        return result

    @staticmethod
    def process_group(group):
        """Process a single group for NMS."""
        BBOX = group[['normxmin', 'normymin', 'normxmax', 'normymax']].values
        SCORE = torch.tensor(group['score'].values)
        EPSILON = group['epsilon'].iloc[0]

        # DBSCAN clustering on bounding boxes
        dbscan = DBSCAN(eps=EPSILON, min_samples=1)
        clusters = dbscan.fit_predict(BBOX)

        # Store indices to keep for this group
        group_loc = []

        # Apply NMS for each cluster
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue  # Skip noise points

            # Get indices of the current cluster
            cluster_indices = (clusters == cluster_id)
            bbox = torch.tensor(BBOX[cluster_indices], dtype=torch.float32)
            score = SCORE[cluster_indices]

            # Perform NMS
            nms_indices = nms(bbox, score, iou_threshold=0.8)
            nms_indices = nms_indices.numpy()

            # Map back to original DataFrame indices
            original_indices = (
                group.index
                [cluster_indices]
                [nms_indices]
                .tolist()
            )
            group_loc.extend(original_indices)

        return group_loc

    @magic.cached.static.property
    def prompt_nms(self) -> Self:
        """Apply NMS for each prompt for each file in parallel."""
        process_group = self.process_group
        _ = self.ifile, self.iclass
        _ = self['xmin ymin xmax ymax epsilon score'.split()]
        results = [
            process_group(group)
            for _, group in
            self.groupby(['ifile', 'prompt'], observed=True, sort=False)
        ]

        # Flatten the list of results
        loc = np.concatenate(results)
        result = self.loc[loc]
        return result

    @magic.index
    def prompt(self) -> magic[str]:
        ...

    @magic.index
    def file(self) -> magic[str]:
        ...

    @magic.index
    def ilogit(self) -> magic[int]:
        ...

    @magic.index
    def ipred(self):
        ...

    @magic.cached.serialized.property
    def threshold(self) -> float:
        ...

    @cached_property
    def c(self) -> Self:
        loc = self.level == 'c'
        return self.loc[loc].copy()

    @cached_property
    def cs(self) -> Self:
        loc = self.level == 'cs'
        return self.loc[loc].copy()

    @cached_property
    def csa(self) -> Self:
        loc = self.level == 'csa'
        return self.loc[loc].copy()

    @magic.Frame
    def labels(self) -> Labels:
        self.elsa.classes.drop_duplicates()

    @magic.column
    def score_name(self):
        ...

    @magic.column
    def iclass(self) -> magic[str]:
        prompts = self.elsa.prompts
        result = (
            prompts.iclass
            .indexed_on(self.prompt, prompts.natural)
            .values
        )
        return result

    @magic.column
    def score(self):
        ...

    @View
    @magic.portal(View.__call__)
    def view(self):
        ...

    @magic.column
    def vanilla(self) -> magic[str]:
        """
        Represents the "vanilla" prompt, composed of the original labels
        from the labels metadata. For example, "person walking" is
        vanilla, while "human strolling" is not.
        """
        result = (
            self.prompts.vanilla
            .set_axis(self.prompts.natural)
            .loc[self.prompt]
            .values
        )
        return result

    @magic.cached.static.property
    def confidence(self) -> Self:
        """Select only the columns that describe confidence"""
        loc = [
            column
            for column in self.columns
            if column.startswith('scores.')
               or column.startswith('score_')
               and column != 'score_name'
        ]
        result = self.loc[:, loc]
        return result

    @magic.column
    def cardinal(self):
        return (
            self.prompts.cardinal
            .set_axis(self.prompts.natural)
            .loc[self.prompt]
            .values
        )

    @magic.series
    def path(self) -> Series[str]:
        result = (
            self.elsa.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @property
    def means(
            self
    ) -> pd.DataFrame:
        loc = [
            column
            for column in self.columns
            if (
                    column.startswith('scores.')
                    or column.startswith('score')
            )
        ]
        loc += ['iclass']
        _ = self.iclass
        result = (
            self
            .loc[:, loc]
            .select_dtypes(include=[float, int])  # Keeps only numeric columns
            .groupby('iclass', sort=False)
            .mean()
            .sort_values('score', ascending=False)
        )
        _ = self.elsa.prompts.cardinal
        cardinal = (
            self.elsa.prompts
            .drop_duplicates('cardinal')
            .set_index('iclass')
            .cardinal
            .loc[result.index]
        )
        result['cardinal'] = cardinal
        return result
