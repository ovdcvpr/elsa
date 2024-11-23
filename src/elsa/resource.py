from __future__ import annotations

from functools import *
from pandas import Series
from pathlib import Path
from typing import *
from typing import Self

import magicpandas as magic

if False:
    from elsa.root import Elsa
    from elsa.images import Images
    from elsa.truth import Truth
    from elsa.labels import Labels
    from elsa.files.files import Files
    from elsa.synonyms.synonyms import Synonyms
    from elsa.annotation.prompts import Prompts
    from elsa.annotation.stacked import Stacked
    import elsa.root

class Resource(
    magic.Magic
):
    """
    A base class with convenience attributes to be used for other
    frames in the library. These attributes allow for easy access to
    various objects regardless of location in the hierarchy.
    """
    owner: Elsa
    outer: Elsa

    @magic.cached.cmdline.property
    def passed(self) -> Optional[Path, str]:
        """The path passed that will be used to construct the object."""
        return None

    @magic.column.from_options(dtype='category')
    def file(self) -> Series:
        """The file names of all unique images in all resources."""
        result = (
            self.elsa.images.file
            .loc[self.ifile]
            .values
        )
        return result

    @magic.column
    def path(self) -> Series[str]:
        result = (
            self.outer.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...

    @magic.index
    def ifile(self) -> magic[str]:
        images = self.elsa.images
        _ = images.ifile
        result = (
            images
            .reset_index()
            .ifile
            .indexed_on(self.file, fill_value='')
            .values
        )
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        raise NotImplementedError

    @cached_property
    def images(self) -> Images:
        elsa = self.elsa
        return elsa.images

    @cached_property
    def truth(self) -> Truth:
        return self.elsa.truth

    @cached_property
    def labels(self) -> Labels:
        return self.elsa.labels

    @cached_property
    def files(self) -> Files:
        return self.elsa.files

    @cached_property
    def synonyms(self) -> Synonyms:
        return self.elsa.synonyms

    @cached_property
    def prompts(self) -> Prompts:
        return self.elsa.prompts

    @cached_property
    def stacked(self) -> Stacked:
        return self.elsa.truth.unique.stacked

    @magic.column
    def source(self):
        return ''

    @magic.column
    def nfile(self) -> magic[int]:
        """An integer [0, N] for each unique ifile"""
        result = (
            self.elsa.images.nfile
            .loc[self.ifile]
            .values
        )
        return result

