from __future__ import annotations
from pathlib import Path
import os
import pandas as pd

import tempfile
from pathlib import Path
from typing import *
from typing import Self

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.annotation.consumed import Consumed
    import elsa.annotation.limit_synonyms
    import elsa.annotation.stacked


class Prompts(
    Resource,
    magic.Frame
):
    """For each stacked, aggregate the labels into a sentence"""
    unique: Self
    outer: Consumed

    def conjure(self) -> Self:
        consumed = self.outer
        loc = ~consumed.iprompt.duplicated()
        columns = 'ilabels isyns'.split()
        result = (
            consumed
            .loc[loc, columns]
            .pipe(self.enchant)
        )
        _ = result.natural, result.catchars, result.ilabels, result.isyns
        loc = ~result.natural.duplicated()
        result = result.loc[loc]
        return result

    @magic.column
    def ilabels(self) -> magic[int]:
        """Identifier for each unique combination of ilabel"""
        result = self.outer.ilabels.loc[self.iprompt].values
        return result

    @magic.column
    def iclass(self) -> magic[int]:
        result = self.elsa.classes.iclass.loc[self.ilabels].values
        return result

    @magic.column
    def is_vanilla(self) -> magic[bool]:
        stacked = self.stacked
        _ = stacked.is_vanilla, stacked.isyns
        result: pd.Series = (
            stacked
            .groupby('isyns', sort=False, observed=True)
            .is_vanilla
            .all()
            .loc[self.isyns]
        )
        self.natural.loc[result.values]
        (
            result
            .groupby(self.iclass.values)
            .sum()
            .max()
        )
        # assert (
        #            result
        #            .groupby(self.iclass.values)
        #            .sum()
        #            .max()
        #        ) == 1
        result = result.values
        return result

    @magic.column
    def vanilla(self) -> magic[str]:
        result = (
            self
            .sort_values('is_vanilla')
            .groupby('iclass', sort=False)
            .natural
            .first()
            .loc[self.iclass]
            .values
        )
        return result

    @magic.index
    # @magic.column
    def iprompt(self) -> magic[int]:
        """Identifier for each stacked for each unique combination of ilabel"""

    @magic.column
    def natural(self) -> Series[str]:
        """ The natural-language prompt """
        stacked = self.stacked
        _ = stacked.iorder, stacked.natural

        result = (
            stacked
            .reset_index()
            .groupby('ilabels iprompt iorder'.split(), sort=False, observed=True)
            .prompt
            .apply(' and '.join)
            .groupby('ilabels iprompt'.split(), sort=False, observed=True)
            .apply(' '.join)
            .droplevel('ilabels')
            .astype('category')
            .loc[self.iprompt]
            .values
        )
        return result

    @magic.column
    def isyns(self) -> Series[str]:
        """A tuple of the unique synonyms present in the synonymous prompt"""

    @magic.column
    def catchars(self) -> Series[str]:
        """The catchars language"""
        stacked = self.outer
        _ = stacked.iorder, stacked.catchars

        result = (
            stacked
            .reset_index()
            .groupby('ilabels iprompt iorder'.split(), sort=False, observed=True)
            .catchars
            .apply('     '.join)
            .groupby('ilabels iprompt'.split(), sort=False, observed=True)
            .apply(' '.join)
            .droplevel('ilabels')
            .loc[self.iprompt]
            .values
        )

        return result

    @magic.column
    def labelchars(self) -> Series[str]:
        """The labelchars language"""
        stacked = self.outer
        _ = stacked.iorder, stacked.labelchars

        result = (
            stacked
            .reset_index()
            .groupby('ilabels iprompt iorder'.split(), sort=False, observed=True)
            .labelchars
            .apply('     '.join)
            .groupby('ilabels iprompt'.split(), sort=False, observed=True)
            .apply(' '.join)
            .droplevel('ilabels')
            .loc[self.iprompt]
            .values
        )

        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary prompt to represent synonymous prompts"""
        result = (
            self.natural
            .groupby(self.ilabels.values)
            .first()
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary prompt to represent synonymous prompts"""
        _ = self.natural, self.ilabels, self.iclass

        result = (
            self
            .sort_values('natural')
            .groupby('iclass', sort=False)
            .natural
            .first()
            .loc[self.iclass]
            .values
        )
        return result


    @magic.column
    def is_in_coco(self):
        """All the labels included in the prompt are in the COCO labels"""
        result = (
            self.outer.is_in_coco
            .groupby(level='ilabels', sort=False)
            .all()
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def is_like_coco(self):
        """All of the labels included in the prompt are synonymous with COCO labels"""
        result = (
            self.outer.is_like_coco
            .groupby(level='ilabels', sort=False)
            .all()
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def combo(self):
        stacked = self.outer
        _ = stacked.ilabel, stacked.label
        result = (
            stacked
            .sort_values('ilabel')
            .groupby(level='iprompt', sort=False)
            .label
            .apply(' '.join)
            .loc[self.iprompt]
            .values
        )
        return result

    @magic.column
    def is_in_truth(self) -> magic[bool]:
        """Which prompts are located in the ground truth annotations"""
        result = self.combo.isin(self.truth.combo).values
        return result

    def implicated(
            self,
            by: Union[DataFrame, Series, str, list, tuple],
            synonyms: bool = True,
    ) -> Series[bool]:
        """
        Determine which prompts are implicated by the frame (e.g. files)
        prompts.implicated_by(files)
        prompts.implicated_by([file, path/to/file.png]
        """

        if synonyms:
            if isinstance(by, (str, tuple)):
                by = [by]
            if isinstance(by, list):
                # files passed
                truth = self.truth
                loc = truth.file.isin(by)
                loc |= truth.path.isin(by)
                loc |= truth.combo.isin(by)
                loc |= truth.natural.isin(by)
                loc |= truth.path.isin(by)
                loc |= truth.file.isin(by)
                ilabels = truth.ilabels.loc[loc].values
                loc = self.ilabels.isin(ilabels)
            elif hasattr(by, 'ilabels'):
                loc = self.ilabels.isin(by.ilabels)
            elif hasattr(by, 'file'):
                file = by.file
                truth = self.truth
                loc = truth.file.isin(file.values)
                ilabels = truth.ilabels.loc[loc].values
                loc = self.ilabels.isin(ilabels)
            elif isinstance(by, Series):
                loc = self.ilabels.isin(by)
            else:
                raise NotImplementedError
        else:
            if hasattr(by, 'combo'):
                loc = self.combo.isin(by.combo)
            elif hasattr(by, 'file'):
                file = by.file
                truth = self.truth
                loc = truth.file.isin(file.values)
                loc = (
                    self.combo
                    .isin(truth.combo.loc[loc])
                )
            elif isinstance(by, Series):
                loc = self.combo.isin(by)
            else:
                raise NotImplementedError
        return loc

    @magic.column
    def file(self):
        raise AttributeError

    @magic.column
    def level(self) -> magic[str]:
        combos = self.truth.combos
        _ = combos.level
        loc = ~combos.ilabels.duplicated()
        level = (
            combos
            .loc[loc]
            .set_index('ilabels')
            .loc[self.ilabels, 'level']
            .values
        )
        return level

    def includes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo contains a label"""
        stacked = self.outer
        if label and cat:
            raise ValueError('label and cat cannot both be provided')
        if label is not None:
            ilabel = self.synonyms.ilabel.from_label(label)
            loc = stacked.ilabel == ilabel
        elif cat is not None:
            loc = stacked.cat == cat
        else:
            raise ValueError('label or cat must be provided')

        result = (
            Series(loc)
            .groupby(stacked.iprompt, sort=False, observed=True)
            .any()
            .loc[self.iprompt]
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo excludes a label"""
        return ~self.includes(label, cat)

    def synonymous_with(self, prompt: str) -> Series[bool]:
        """Determine if a prompt is synonymous"""
        loc = self.natural == prompt
        ilabels = self.ilabels.loc[loc].first()
        loc = self.ilabels == ilabels
        return loc

    def contains_substring(self, substring: str) -> Series[bool]:
        return self.natural.str.contains(substring)

    # def get_nunique_labels(self, loc=None) -> Series[bool]:
    #     """
    #     Pass a mask that is aligned with the annotations;
    #     group that mask by the ilabels and count the number of unique labels
    #     """
    #     if loc is None:
    #         loc = slice(None)
    #     stacked = self.stacked
    #     result = Series(0, index=self.iprompt)
    #     iprompt = stacked.loc[loc].iprompt.values
    #     update = (
    #         stacked.ilabel
    #         .loc[loc]
    #         .groupby(iprompt, sort=False, observed=True)
    #         .nunique()
    #     )
    #     result.update(update)
    #     # result = result.values
    #     return result

    def write(self, file: str = None):
        """Write all unique prompts to a file"""
        if file is None:
            file = Path(tempfile.gettempdir(), 'prompts.txt').resolve()
        prompts: list[str] = self.natural.unique().tolist()
        outpath = Path(__file__, '..', file).resolve()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, 'w+') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        print(outpath)

    @magic.series
    def nprompt(self):
        natural = self.natural.drop_duplicates()
        result = (
            Series(np.arange(len(natural)), index=natural)
            .loc[self.natural]
            .set_axis(self.index)
        )
        return result

    @magic.series
    def natural2level(self) -> Series:
        _ = self.level
        result = (
            self.level
            .set_axis(self.natural)
        )
        return result

    @magic.column
    def condition(self) -> magic[str]:
        """Represents which condition out of {person, pair, people} the prompt has."""
        person = self.includes('person')
        pair = self.includes('pair')
        people = self.includes('people')
        none = ~person & ~pair & ~people
        assert (
                pair
                ^ person
                ^ people
                | none
        ).all()
        result = np.full_like(person, '', dtype=object)
        result = np.where(person, 'person', result)
        result = np.where(pair, 'pair', result)
        result = np.where(people, 'people', result)
        return result

    @magic.series
    def ilabels_string(self) -> Series[str]:
        """
        Tuples are not easily serialized; store them as strings instead.
        """
        result = (
            self.elsa.classes.ilabels_string
            .loc[self.ilabels]
            .values
        )
        return result

    @magic.column
    def is_disjoint(self) -> Series[bool]:
        ...

    limit_synonyms: elsa.annotation.limit_synonyms.LimitSynonyms

    @magic.delayed
    def limit_synonyms(self) -> elsa.annotation.limit_synonyms.LimitSynonyms:
        ...

    # # @magic.cached.static.property
    # def limit_synonyms(self, n) -> Self:
    #     result = (
    #         self
    #         .reset_index()
    #         .groupby('ilabels', sort=False, observed=True)
    #         .head(n)
    #     )
    #     return result

    @magic.cached.outer.property
    def stacked(self) -> elsa.annotation.stacked.Stacked:
        ...

    def in_directory(
            self,
            directory: str | Path,
    ) -> Series[bool]:
        directory = Path(directory)
        names: set[str] = {
            f.stem for f in directory.rglob("*.parquet")
        }
        loc = self.natural.isin(names)
        return loc

    def purge(self, directory: str | Path):
        """
        List all directorys that end with .parquet in the directory.
        If their directorynames are not in any of the natural prompts, delete
        them. If no directorynames are in the natural prompts, raise an Error.
        """
        directory = Path(directory)

        # Ensure the natural series is not empty
        if self.natural.empty:
            raise ValueError("The natural series is empty; no filenames to match.")

        # Convert natural prompts to a set of valid filenames (without extensions)
        natural_filenames = set(self.natural)

        # List all .parquet files in the directory
        parquet_files = list(directory.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in directory: {directory}")

        # Track if we find any matching files
        found_matching_file = False

        # Iterate through all .parquet files
        for parquet_file in parquet_files:
            # Extract the filename without the extension
            filename_without_ext = parquet_file.stem
            if filename_without_ext in natural_filenames:
                found_matching_file = True

        # Raise an error if no filenames were found in the natural prompts
        if not found_matching_file:
            raise ValueError("No filenames matched any of the natural prompts.")
        # Iterate through all .parquet files
        for parquet_file in parquet_files:
            # Extract the filename without the extension
            filename_without_ext = parquet_file.stem

            if filename_without_ext not in natural_filenames:
                # If the file name doesn't match any in the natural prompts, delete it
                parquet_file.unlink()  # This deletes the file
