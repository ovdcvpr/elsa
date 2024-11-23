from __future__ import annotations

import glob
import numpy as np
import os
import pandas as pd
import warnings
from pandas import DataFrame
from pandas import Series
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa import util
from elsa.resource import Resource

if False:
    from elsa.root import Elsa


class ClassProperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


class Files(
    Resource,
    magic.Frame
):
    outer: Elsa
    """
    A DataFrame representing the literal image files available;
    elsa.images contains metadata about the images but elsa.files
    contains all the files in the directory and their actual paths.
    """

    def conjure(self) -> Self:
        """Called when accessing Elsa.files to instantiate Files"""
        elsa = self.outer
        with self.configure:
            passed = self.passed
        result = (
            self
            .from_inferred(passed)
            .pipe(self.enchant)
        )
        # assert len(result), f'No files found in {passed}'
        result.file = util.trim_path(result.file)
        loc = ~result.file.isin(self.owner.images.file)
        if loc.any():
            eg = result.file[loc].iloc[0]
            self.logger.warning(
                f'{loc.sum()} files in {passed} e.g. {eg} are not '
                f'present in the images metadata. These files will '
                f'be ignored.'
            )
            result = result.loc[~loc].copy()
        # _ = result.ifile
        loc = result.ifile.isin(elsa.ifile).values
        # loc = result.ifile.isin(elsa.ifile)
        result = result.loc[loc].copy()
        result = result.set_index('ifile')

        return result

    # def conjure(self) -> Self:
    #     """Called when accessing Elsa.files to instantiate Files"""
    #     elsa = self.outer
    #     images = elsa.images
    #     files = (
    #         images
    #         .reset_index()
    #         ['file ifile download'.split()]
    #         .set_index('file')
    #         .pipe(self.enchant)
    #     )
    #     with self.configure:
    #         passed = self.passed
    #     if isinstance(passed, str):
    #         passed = passed,
    #     for directory in passed:
    #         if not os.path.isdir(directory):
    #             raise ValueError(f'{directory} is not a directory')
    #         paths = [
    #             os.path.join(root, filename)
    #             for root, _, filenames in os.walk(directory)
    #             for filename in filenames
    #         ]
    #         names = [
    #             os.path.basename(path).rsplit('.', 1)[0]
    #             for path in paths
    #         ]
    #         path = Series(paths, index=names, name='path')
    #         files.update(path)
    #     return files

    @magic.column
    def width(self) -> Series[int]:
        """ Width of the image in pixels """
        return self.images.width.loc[self.ifile].values

    @magic.column
    def height(self) -> Series[int]:
        """ Height of the image in pixels """
        return self.images.height.loc[self.ifile].values

    @magic.column
    def path(self) -> Series[str]:
        """The absolute path to each image file"""

    @classmethod
    def from_directory(cls, directory: str | Path) -> Self:
        """Create a Files object from a directory of images"""
        path = os.path.join(directory, f'*.{cls.extension}')
        paths = glob.glob(path)
        file = np.fromiter((
            file
            .rsplit(os.sep, maxsplit=1)[-1]
            .rsplit('.', maxsplit=1)[0]
            for file in paths
        ), dtype=object, count=len(paths))
        result = cls({
            'path': paths,
            'file': file,
        })
        result.passed = directory
        return result

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> Self:
        """Create a Files object from a list of paths"""
        concat = list(map(cls.from_inferred, paths))
        try:
            result = pd.concat(concat)
        except ValueError as e:
            result = cls({
                'ifile': []
            })
        result = (
            result
            .pipe(cls)
            .drop_duplicates('file')
        )
        result.passed = paths
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        """Create a Files object, inferring the procedure from the input"""
        if isinstance(path, (list, tuple)):
            result = cls.from_paths(path)
        elif isinstance(path, (Path, str)):
            result = cls.from_directory(path)
        else:
            msg = f'Elsa.files expected a Path, str, or Iterable, got {type(path)}'
            raise TypeError(msg)
        result.passed = path
        return result

    @magic.column
    def nboxes(self) -> magic[int]:
        """How many boxes in the truth belong to this file"""
        result = (
            self.truth.combos
            .groupby('ifile')
            .size()
            .reindex(self.ifile, fill_value=0)
            .values
        )

        return result

    def includes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """True where a combo contains a label"""
        ann = self.elsa.truth
        if label and cat:
            raise ValueError('label and cat cannot both be provided')
        if label is not None:
            ilabel = self.synonyms.ilabel.from_label(label)
            loc = ann.ilabel == ilabel
        elif cat is not None:
            loc = ann.cat == cat
            loc |= ann.cat_char == cat
        else:
            raise ValueError('label or cat must be provided')

        result = (
            Series(loc)
            .groupby(ann.file, sort=False)
            .any()
            # .loc[truth.ibox]
        )
        return result


    # todo but prompts is a resource
    def implicated(
            self,
            by: Union[DataFrame, Series, str, list]
    ) -> Series[bool]:
        """
        Determine which files are implicated by the farme (e.g. prompts)
        files.implicated_by(prompts)
        files.implicated_by((person walking, a person sitting on a chair))
        files.implicated_by(((0,5,7), (3,27,50))
        """
        if isinstance(by, (str, tuple)):
            by = [by]
        if isinstance(by, list):
            # select files that contain any of the prompts
            prompts = self.elsa.prompts
            loc = prompts.combo.isin(by)
            loc |= prompts.natural.isin(by)
            loc |= prompts.isyns.isin(by)
            isyns = prompts.isyns.loc[loc].values
            loc = self.truth.isyns.isin(isyns)
            file = self.truth.file.loc[loc].values
            loc = self.file.isin(file)
            loc |= self.file.isin(by)
            loc |= self.path.isin(by)
        elif 'file' in by:
            # select files that are implicated by the frame
            loc = self.file.isin(by.file)
        elif 'isyns' in by:
            # select files that contain synonyms to the frame
            truth = self.truth
            loc = truth.isyns.isin(by.isyns)
            file = truth.file.loc[loc].values
            loc = self.file.isin(file)
        else:
            raise NotImplementedError
        return loc

        # if hasattr(frame, 'file'):
        #     loc = self.file.isin(frame.file)
        # else:
        #     raise NotImplementedError
        # return loc

    # todo: should be magic cached property
    extension: str = 'png'

#     def includes(
#             self,
#             label: str = None,
#             cat: str = None,
#     ) -> Series[bool]:
#         truth = self.elsa.truth
#         loc = truth.includes(label, cat)
#         loc = loc.index[loc]
#         self.ifile.isin(truth.ifile.loc[loc])
#
#     def excludes(
#             self,
#             label: str = None,
#             cat: str = None,
#     ) -> Series[bool]:
#         return ~self.includes(label, cat)
#
#
# from pandas.core.indexing import _LocIndexer
# _LocIndexer.__call__
