from __future__ import annotations

import os
from pathlib import Path
from typing import *

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only log errors

# otherwise we get these warnings every time we import elsa
# 2024-07-24 00:36:07.033491: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-07-24 00:36:07.056229: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2024-07-24 00:36:07.432384: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

import elsa.files.files
import elsa.images
import elsa.labels
import elsa.truth.truth
import magicpandas as magic
from elsa.classes.classes import Classes
from elsa.files.files import Files
from elsa.images import Images
from elsa.disjoint import Disjoint
from elsa.labels import Labels
from elsa.local import local
from elsa.predict.predict import Predict
from elsa.prediction.prediction import Prediction
from elsa.scored.scored import Scored
from elsa.resource import Resource
from elsa.synonyms.synonyms import Synonyms
from elsa.truth.truth import Truth
from elsa.annotation.consumed import Prompts
import elsa.annotation.consumed
import elsa.annotation.stacked
from elsa.report import Report
from elsa.scored.cdba.cdba import CDBA
from elsa.scored.nms.nms import NMS
from elsa.scored.normal.normal import Normal
from elsa.ven import Ven

pd.options.mode.chained_assignment = None


class Elsa(
    Resource,
    magic.Frame
):
    """
    Root instance for the project, encapsulating all resources into
    one file for easy access and manipulation.
    """

    @Classes
    @magic.portal(Classes.conjure)
    def classes(self):
        """
        A DataFrame representing all unique classes implicated by the
        ground truth.
        """

    @Files
    @magic.portal(Files.conjure)
    def files(self):
        """
        A DataFrame representing the literal image files available;
        elsa.images contains metadata about the images but elsa.files
        contains all the files in the directory and their actual paths.
        """

    @Images
    @magic.portal(Images.conjure)
    def images(self):
        """
        A DataFrame of all Images, mapping filenames to paths,
        image sizes, and other metadata.
        """

    @Disjoint
    @magic.portal(Disjoint.conjure)
    def disjoint(self):
        """
        A DataFrame representing all possible reasons that any
        annotation combinations in the dataset may be disjoint. For
        example, "person standing sitting" is disjoint: a person cannot
        be both standing and sitting.
        """

    @Labels
    @magic.portal(Labels.conjure)
    def labels(self):
        """
        A DataFrame of the unique Labels used by the dataset, mapping
        their names to their IDs,
        """

    if False:
        magic.portal(Predict.gdino.batched.__call__)

    @Predict
    # @magic.portal(Predict.gdino.batched.__call__)
    def predict(self):
        """
        The predict module allows the user to run inference with
        different models for the images and prompts represented by this
        instance.

        For example:
        elsa.predict(...)
        elsa.predict.gdino.batched(...)
        elsa.predict.mdetr.batched(...)
        """

    @Prediction
    @magic.portal(Prediction.__call__)
    def prediction(self):
        """
        When prediction is performed, it serializes the results into a
        directory, with each batch of predictions saved in a file named
        by its prompt. This module wraps each individual prediction,
        allowing for micro-level analysis of the results:

        elsa.prediction('person walking.parquet')
        """

    @Scored
    @magic.portal(Scored.__call__)
    def scored(self):
        """
        To perform macroscopic analysis of a model's predictions we must
        concatenate the results of all predictions into a single
        file, thresholded by a particular score. This module contains
        metrics to evaluate the performance of the model's inference.

        elsa.scored(
            file='gdino.selected.nlse>.3.parquet',
            directory='/predictions',
            score='selected.nlse',
            threshold=.3
        )
        """

    @Synonyms
    @magic.portal(Synonyms.conjure)
    def synonyms(self):
        """
        A DataFrame representing which labels are synonymous, and other
        metadata such as their category (condition, state, activity,
        other), natural representation (person -> a person), and whether
        these synonyms are used in the prompt generation.
        """

    @Truth
    @magic.portal(Truth.conjure)
    def truth(self):
        """
        A DataFrame encapsulating the ground truth annotations from the
        dataset, containing the bounding boxes and their assigned labels.
        """

    @magic.cached.static.property
    @magic.portal(Prompts.conjure)
    def prompts(self) -> Prompts:
        """
        A DataFrame containing the prompts generated by the unique
        combinations of synonymous labels for each class.
        For example, the class "person walking" may have two sets of synonyms:

        person:
            person
            individual
        walking:
            walking
            strolling

        The prompts module will generate the following prompts:
            person walking
            person strolling
            individual walking
            individual strolling
        """
        return self.truth.unique.stacked.consumed.prompts

    @Report
    @magic.portal(Report.__call__)
    @magic.portal(CDBA.__call__)
    @magic.portal(NMS.__call__)
    @magic.portal(Normal.__call__)
    def report(self):
        """
        A module to generate a report with the various metrics, such as
        AP, mAP, TP counts, total counts, and other metrics for the sets
        of predictions run. For example:

        elsa.report(
            # pass as many scored files as you like;
            # these are the files that are saved from calling
            # elsa.scored(file=..., directory=...)
            'gdino.parquet',
            'mdetr.parquet',

            # whether to overwrite previous CSV reports for each scored file
            overwrite=True,

            outdir='./report',
        )

        files:
            List of concatencated parquet files to score, or
            directories containing parquet files to score. See
            `Elsa.scored` to how to generate these.
        outdir:
            The output directory in which a subdirectory for each file
            will be created. For example, passing `gdino_nlse.parquet`
            results in a gdino_nlse directory.
        overwrite:
            if True, existing CSV files will be overwritten.
        metrics:
            A list of which metrics to report on. For each metric,
            CSVs are generated, and then at the end of the iteration,
            those CSVs are concatenated.
        method_parameters:
            A dictionary of method parameters. The key is the method
            name, and the value is a dictionary of parameters to pass
            to the method. For example, to change CDBA IOU threshold,
            pass `{'cdba': {'iou': .7}}`.
        """

    @Ven
    @magic.portal(Ven.__call__)
    def ven(self):
        """
        Compare the outcomes of the postprocessing methods DBA and NMS
        on scored predictions. The table generated is essentially a
        "ven diagram"; you can determine the count of differing true
        positives and shared true positives between the two methods.
        """

    with magic.default:
        default = magic.default(
            images=images.passed,
            labels=labels.passed,
            truth=truth.passed,
            files=files.passed,
        )

    @classmethod
    @default
    def from_resources(
            cls,
            truth: str | Path = None,
            images: str | Path = None,
            labels: str | Path = None,
            files: str | Path = None,
            quiet: bool = False,
    ) -> Self:
        """
        Instantiate Raster instance from the specified resource paths.

        truth:
            str | Path:
                file or directory used to generate elsa.truth
        images:
            str | Path:
                file used to generate elsa.images
            dict[str, str | Path]:
                mapping of source name to image metadata file, e.g:
                {'BSV_': ..., 'GSV_': ...}
        labels:
            str | Path:
                file used to generate elsa.labels
        files:
            str | Path:
                path to the directory that contains the literal image files
        quiet:
            bool:
                silence warnings during construction
        """
        elsa = cls()
        elsa.images = (
            Images
            .from_inferred(images)
            .assign(elsa=elsa)
        )
        elsa.truth = (
            Truth
            .from_inferred(truth)
            .assign(elsa=elsa)
        )
        elsa.files = (
            Files
            .from_inferred(files)
            .assign(elsa=elsa)
        )
        # drop truth not in images
        loc = ~elsa.truth.ifile.isin(elsa.images.ifile)
        if loc.any():
            eg = (
                elsa.truth.ifile
                .loc[loc]
                .drop_duplicates()
                .tolist()
            )
            msg = (
                f'{loc.sum()} files out of {len(loc)} in the gt '
                f'annotations e.g. {eg} are not in the image metadata '
                f'and are being dropped.'
            )
            if not quiet:
                elsa.logger.info(msg)
            # elsa.logger.info(msg)
        if len(elsa.files):
            loc = ~elsa.files.file.isin(elsa.images.file)
            if loc.any():
                eg = (
                    elsa.files.file
                    .loc[loc]
                    .drop_duplicates()
                    .tolist()
                    [:10]
                )
                msg = (
                    f'{loc.sum()} files out of {len(loc)} in the files '
                    f'metadata e.g. {eg} are not in the image metadata and '
                    f'are being dropped.'
                )
                # elsa.logger.info(msg)
                if not quiet:
                    elsa.logger.info(msg)

        index = (
            pd.Index(elsa.truth.ifile)
            .intersection(elsa.images.ifile)
            # .intersection(elsa.files.ifile)
            # .unique()
            # .sort_values()
        )
        if len(elsa.files):
            index = index.intersection(elsa.files.ifile)
        index = index.unique().sort_values()

        result = cls(index=index)
        level = result.logger.level
        if quiet:
            result.logger.setLevel('ERROR')
        with result.configure:
            result.truth.passed = truth
            result.images.passed = images
            result.labels.passed = labels
            if isinstance(files, (str, Path)):
                result.files.passed = (
                    Path(files)
                    .expanduser()
                    .resolve()
                )
            elif isinstance(files, (list, tuple)):
                result.files.passed = [
                    Path(v)
                    .expanduser()
                    .resolve()
                    for v in files
                ]
            # result.files.passed = files

        # drop files not in images or images not in files
        images = result.images
        assert images.__order__ == 3
        files = result.files
        if len(files):
            ifile = images.ifile.intersection(files.ifile)

            # drop images not in files
            loc = images.ifile.isin(ifile)
            total = len(images)
            dropped = (~loc).sum()
            if dropped:
                eg = (
                    images.ifile
                    [~loc]
                    .drop_duplicates()
                    .tolist()
                    [:10]
                )
                msg = (
                    f'{dropped} files e.g. {eg} in the images metadata out '
                    f'of {total} are not in the literal image files and are '
                    f'being dropped.'
                )
                # result.logger.info(msg)
                if not quiet:
                    result.logger.info(msg)
                images = images.loc[loc]

            # drop files not in images
            loc = files.ifile.isin(ifile)
            total = len(files)
            dropped = (~loc).sum()
            if dropped:
                eg = (
                    files.ifile
                    .loc[~loc]
                    .drop_duplicates()
                    .tolist()
                    [:10]
                )
                msg = (
                    f'{dropped} files e.g. {eg} in the files metadata out of'
                    f' {total} are not in the images metadata and are being '
                    f'dropped.'
                )
                if not quiet:
                    result.logger.info(msg)
                files = files.loc[loc]

        result.files = files
        result.images = images

        # todo: result.truth is dependeng on files, but has not yet been
        #   updated to reflect the changes in files

        # drop truth not in files, if files was passed
        truth = result.truth
        files = result.files
        if len(files):
            ifile = files.ifile.intersection(truth.ifile)
            loc = truth.ifile.isin(ifile)
            total = truth.ifile.nunique()
            dropped = truth.ifile.loc[~loc].nunique()
            if dropped:
                eg = (
                    truth.ifile
                    .loc[~loc]
                    .drop_duplicates()
                    .tolist()
                    [:10]
                )
                msg = (
                    f'{dropped} files e.g. {eg} in the truth metadata out of'
                    f' {total} are not in the files metadata and are being '
                    f'dropped.'
                )
                truth = truth.loc[loc]
                if not quiet:
                    result.logger.info(msg)

        result.logger.setLevel(level)
        truth = truth.copy()

        needles = (
            truth['ibox ilabel'.split()]
            .pipe(pd.MultiIndex.from_frame)
        )
        loc = needles.duplicated()
        if loc.any():
            eg = (
                truth.ibox
                .loc[loc]
                .tolist()
                [:10]
            )
            msg = (
                f'Dropping {loc.sum()} annotations from the truth '
                f'because they are duplicated in ibox={eg}'
            )
            if not quiet:
                result.logger.warn(msg)
            truth = truth.loc[~loc]

        result.truth = truth
        return result

    @classmethod
    def from_google(
            cls,
            truth: str | Path = None,
            images: str | Path = None,
            labels: str | Path = None,
            files: str | Path = None,
            quiet: bool = False,
    ) -> Self:
        """Instantiate Elsa specific to the Google dataset."""
        result = cls.from_resources(
            # images=images or elsa.images.google,
            images=images or elsa.images.unified,
            # labels=labels or elsa.labels.google,
            labels=labels or elsa.labels.unified,
            # truth=truth or elsa.truth.truth.google,
            truth=truth or elsa.truth.truth.unified,
            files=files or local.files.google,
            quiet=quiet,
        )
        return result

    @classmethod
    def from_bing(
            cls,
            truth: str | Path = None,
            images: str | Path = None,
            labels: str | Path = None,
            files: str | Path = None,
            quiet: bool = False,
    ) -> Self:
        """Instantiate Elsa specific to the Bing dataset."""
        result = cls.from_resources(
            # images=images or elsa.images.bing,
            images=images or elsa.images.unified,
            # labels=labels or elsa.labels.bing,
            # truth=truth or elsa.truth.truth.bing,
            labels=labels or elsa.labels.unified,
            truth=truth or elsa.truth.truth.unified,
            files=files or local.files.bing,
            quiet=quiet,
        )
        return result

    @classmethod
    def from_unified(
            cls,
            truth: str | Path = None,
            images: str | Path = None,
            labels: str | Path = None,
            files: str | Path = None,
            quiet: bool = False,
    ) -> Self:
        """Instantiate Elsa that includes both Google and Bing datasets."""
        result = cls.from_resources(
            images=images or elsa.images.unified,
            labels=labels or elsa.labels.unified,
            truth=truth or elsa.truth.truth.unified,
            files=files or local.files.unified,
            quiet=quiet,
        )
        return result

    @property
    def file(self):
        """"""
        raise NotImplementedError

    @magic.series
    def cat2char(self) -> magic[str]:
        """Map categoryes to representative characters"""
        result = pd.Series({
            'condition': 'c',
            'state': 's',
            'activity': 'a',
            'others': 'o',
            '': ' ',
        })
        return result
