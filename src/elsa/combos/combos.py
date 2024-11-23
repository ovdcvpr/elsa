from __future__ import annotations

from functools import cached_property
from typing import *
from typing import Self

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pandas import Series

import elsa.util as util
import magicpandas as magic
from elsa.boxes import Boxes
from elsa.classes.has import ILabels
from elsa.combos.disjoint import Disjoint
from elsa.truth.has import IAnn

if False:
    from elsa.annotation import Annotation

class Combos(
    Boxes,
    ILabels,
    IAnn,
):
    outer: Annotation

    def conjure(self) -> Self:
        """
        Called when accessing Annotation.combos to instantiate Combos
        """
        anns = self.outer
        loc = (
            ~anns .ibox
            .duplicated()
            .values
        )
        columns = anns.columns.intersection(self.__defined_columns__)
        result = (
            anns
            .loc[loc, columns]
            .reset_index()
            .set_index('ibox')
            .pipe(self.enchant)
        )
        del result.label
        _ = result.label
        return result


    @magic.index
    def ibox(self) -> Series[int]:
        """Unique index for each combo entry"""


    @magic.column
    def label(self) -> Series[str]:
        """Multilabel from the joined labels"""
        anns = self.outer
        label = (
            anns
            .drop_duplicates('ibox ilabel'.split())
            .sort_values('ilabel')
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
            .loc[self.ibox]
            .values
        )
        return label

    @magic.column
    def iclass(self) -> magic[int]:
        result = (
            self.elsa.classes.iclass
            .loc[self.ilabels]
            .values
        )
        return result

    @Disjoint
    def disjoint(self):
        """
        A DataFrame that broadcasts Elsa.disjoint to each combo in the
        annotations, representing which annotations are disjoint and why.
        """

    @magic.column
    def is_disjoint(self) -> Series[bool]:
        """
        Whether the boxes are disjoint.
        disjoint boxes are considered false positives.
        """
        unique = self.outer.unique.consumed
        disjoint = unique.disjoint.all_checks.any(axis=1)
        ilabels = disjoint[disjoint].index
        ilabels = unique.ilabels.loc[ilabels]
        loc = self.ilabels.isin(ilabels)
        return loc

    @magic.column
    def nlabels(self) -> Series[int]:
        """How many labels are in the combo"""
        result = (
            self.outer
            .groupby('ibox', sort=False)
            .size()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def nunique_labels(self):
        """How many unique labels are in the combo"""
        outer = self.outer
        result = (
            outer.ilabel
            .groupby(outer.ibox, sort=False)
            .nunique()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.cached.cmdline.property
    def threshold(self) -> float:
        """iou threshold"""
        return .9


    @magic.column
    def rcombo(self) -> Series[int]:
        """
        The relative index of each combo for that
        file; the lowest index is 0 for each file e.g.
        0 1 2 3 0 1 0 1 2 0 0 rcombo
        a a a a b b c c c d d file
        """
        arrays = self.ifile, self.ibox
        names = 'ifile ibox'.split()
        needles = pd.MultiIndex.from_arrays(arrays, names=names)
        haystack = needles.unique().sort_values()
        ifile = haystack.get_level_values('ifile')
        con = util.constituents(ifile)
        ifirst = con.ifirst.repeat(con.repeat)
        rcombo = np.arange(len(haystack)) - ifirst
        result = (
            Series(rcombo, index=haystack)
            .loc[needles]
            .values
        )
        return result

    @magic.column
    def color(self) -> Series[str]:
        """The color for each box"""
        rcombo = self.rcombo
        assert rcombo.max() < len(util.colors)
        result = (
            Series(util.colors)
            .iloc[rcombo]
            .values
        )
        return result

    def print(
            self,
            ifile: str | int = None,
            columns: str = tuple('color label'.split()),
    ):
        # todo: also add level, ilevel, ilabel
        for col in columns:
            getattr(self, col)
        columns = list(columns)
        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
        ):
            sub = self.at_ifile(ifile)
            print(sub[columns])

    def view(
            self,
            ibox: Union[int, List[int]] = None,
            file: str = None,
            colors=None,
            background='black',
    ) -> Image:
        if (
                file is None
                and ibox is None
        ):
            file = self.file.values[0]
            loc = self.file == file
            ibox = self.ibox[loc]
        elif ibox is None:
            loc = self.file == file
            loc |= self.ifile == file
            loc |= self.path == file
            ibox = self.ibox[loc]
        elif file is None:
            if isinstance(ibox, int):
                ibox = [ibox]
        else:
            raise ValueError('Must provide file or ibox, not both')
        ifile = self.ifile.loc[ibox].values[0]
        file = self.file.loc[ibox].values[0]
        path = self.elsa.files.path.loc[ifile]

        iloc = self.ibox.get_indexer(ibox)
        assert len(set(self.iloc[iloc].file)) <= 1, "All iboxes must belong to the same file"
        c = self
        _ = c.xmin, c.ymin, c.xmax, c.ymax, c.path, c.label
        combos = self.iloc[iloc]

        image = Image.open(path).convert('RGBA')
        # Create a new image with extra space for text
        width, height = image.size
        new_width = width + 300
        new_image = Image.new('RGBA', (new_width, height), 'white' if background == 'white' else 'black')
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        font = util.font
        header_text_color = 'black' if background == 'white' else 'white'
        util.draw_text_with_outline(draw, (width + 10, 10), f'file={file}', font, header_text_color, 'black', 1)
        y_offset = 50
        colors = util.colors

        for i, combo in enumerate(combos.itertuples()):
            color = colors[i % len(colors)]
            # xy = combo.w, combo.s, combo.e, combo.n
            xy = combo.xmin, combo.ymin, combo.xmax, combo.ymax
            draw.rectangle(xy, outline=color, width=3)
            text_truth = f'truth={combo.label}'
            util.draw_text_with_outline(draw, (width + 10, y_offset), text_truth, font, color, 'black', 1)
            y_offset += 20

        return new_image

    def views(
            self,
            loc: slice = None,
            ibox: bool = False,
            file: bool = False,
            background='black',
    ):
        """
        Given an optional mask, iteratively visualize each image
        that matches the mask.
        """
        if file:
            if loc is None:
                loc = slice(None)
            ifiles = self.ifile[loc].unique()
            for file in ifiles:
                yield self.view(file=file, background=background)
        elif ibox:
            if loc is None:
                loc = slice(None)
            iboxes = self.ibox[loc]
            for ibox in iboxes:
                yield self.view(ibox=ibox, background=background)
        else:
            if loc is None:
                loc = slice(None)
            iboxes: Series = self.ibox[loc]
            file: Series = self.ifile.loc[loc]
            for ibox in (
                    iboxes
                            .groupby(file)
                            .values()
            ):
                yield self.view(ibox=ibox, background=background)

    def at_ifile(self, ifile: str = None) -> Self:
        """ Return a view of the combos at a given file """
        if ifile is None:
            ifile = self.ifile.values[0]
        loc = self.ifile == ifile
        if not loc.any():
            loc = self.file == ifile
            if not loc.any():
                raise ValueError(f'No boxes for {ifile=}')
        self = self.loc[loc]
        return self

    def at_file(self, file: str) -> Self:
        """ Return a view of the combos at a given file """
        if file is None:
            file = self.file.values[0]
        loc = self.file == file
        if not loc.any():
            raise ValueError(f'No boxes for {file=}')
        self = self.loc[loc]
        return self

    # self.synonyms.ilabel.given(string)

    def includes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """True where a combo contains a label"""
        ann = self.outer
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
            .groupby(ann.ibox, sort=False)
            .any()
            .loc[self.ibox]
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            cat: str | int = None,
    ) -> Series[bool]:
        """True where a combo excludes a label"""
        return ~self.includes(label, cat)

    def contains_substring(self, substring: str) -> Series[bool]:
        return self.label.str.contains(substring)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Given a mask that is aligned with the annotations,
        determine the number of unique labels in each combo
        """
        if loc is None:
            loc = slice(None)
        ann = self.outer
        result = Series(0, index=self.ibox)
        ibox = ann.ibox[loc].values
        update = (
            ann.ilabel
            .loc[loc]
            .groupby(ibox)
            .nunique()
        )
        result.update(update)
        result = result.set_axis(self.index)
        return result

    @magic.column
    def ilabels(self) -> Series[tuple[int]]:
        """An ordered tuple of the isyns associated with the combo"""

        # use truth.combos' categorical dtype
        ann = self.outer
        _ = ann.ibox, ann.ilabel
        result = (
            ann
            .reset_index()
            .drop_duplicates('ibox ilabel'.split())
            .sort_values('ilabel')
            .groupby('ibox', sort=False)
            .ilabel
            .apply(tuple)
        )

        if self.outer.__second__ is self.elsa.truth.__second__:
            dtype = result.unique().tolist()
            dtype.append(tuple())
            sort = sorted(dtype)
            dtype = pd.CategoricalDtype(sort)
        else:
            dtype = self.elsa.truth.combos.ilabels.dtype

        result = (
            result
            .astype(dtype)
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def c(self) -> Series[str]:
        """
        The combo label with only the condition;
        None if no condition is present
        """
        result = Series('', index=self.ibox, dtype=object)
        ann = self.outer
        loc = ann.cat.values == 'condition'
        loc &= (
            self.includes(cat='condition')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def cs(self) -> Series[str]:
        """
        The combo label with only the condition and state;
        '' if condition or state are not present
        """
        result = Series('', index=self.ibox, dtype=object)
        ann = self.outer
        loc = ann.cat.values == 'condition'
        loc |= ann.cat.values == 'state'
        loc &= (
            self.includes(cat='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='state')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def csa(self) -> Series[str]:
        """
        The combo label with only the condition, state, and activity;
        '' if condition, state, or activity are not present
        """
        result = Series('', index=self.ibox, dtype=object)
        ann = self.outer
        loc = ann.cat.values == 'condition'
        loc |= ann.cat.values == 'state'
        loc |= ann.cat.values == 'activity'
        loc &= (
            self.includes(cat='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='state')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='activity')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def csao(self) -> Series[str]:
        """
        The combo label with the condition, state, activity, and others;
        '' if condition, state, activity, or others is not present
        """
        result = Series('', index=self.ibox, dtype=object)
        ann = self.outer
        loc = ann.cat.values == 'condition'
        loc |= ann.cat.values == 'state'
        loc |= ann.cat.values == 'activity'
        loc |= ann.cat.values == 'others'
        loc &= (
            self.includes(cat='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='state')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='activity')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(cat='others')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @cached_property
    def annotations(self) -> Annotation:
        return self.outer

    @magic.column
    def cardinal(self) -> magic[str]:
        ...

    @magic.Frame
    def matrix(self):
        result = (
            self.outer.matrix.unique
            .loc[self.ibox]
        )
        return result

    @magic.column
    def condition(self):
        result = (
            self.elsa.classes.condition
            .loc[self.ilabels.values]
            .values
        )
        return result
