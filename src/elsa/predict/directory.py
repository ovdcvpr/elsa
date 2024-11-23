from __future__ import annotations
from pathlib import Path

import os.path

import numpy as np
import pandas as pd
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa.scored.scored  import Scored
from elsa.annotation.prompts import Prompts

if False:
    import elsa.root
    from elsa.root import Prompts


class Directory(magic.Frame):
    @magic.index
    def prompt(self) -> magic[str]:
        ...

    @magic.column
    def outpath(self) -> magic[str]:
        ...

    @magic.cached.outer.property
    def elsa(self) -> elsa.root.Elsa:
        ...

    @magic.column
    def exists(self):
        result = np.fromiter(
            map(os.path.exists, self.outpath),
            dtype=bool,
            count=len(self.outpath),
        )
        return result

    @magic.column
    def iprompt(self):
        prompts = self.elsa.prompts
        result = (
            pd.Series(prompts.iprompt, index=prompts.natural)
            .loc[self.prompt]
            .values
        )
        return result

    @magic.cached.static.property
    def prompts(self) -> Prompts:
        return (
            self.elsa.prompts
            .loc[self.iprompt]
            .copy()
        )

    # noinspection PyTypeHints,PyMethodOverriding
    def __call__(
            self,
            outdir: str,
            prompts=None,
            *args,
            **kwargs
    ) -> Self:
        PROMPTS = prompts
        _ = PROMPTS.natural, PROMPTS.cardinal
        if prompts is None:
            prompts = slice(None)
        elif isinstance(prompts, int):
            loc = np.full_like(PROMPTS.natural, False)
            loc[:prompts] = True
            prompts = loc
        elif isinstance(prompts, Prompts):
            ...
        else:
            prompts = PROMPTS.loc[prompts]
        if not isinstance(prompts, Prompts):
            raise TypeError()
        # prompts = prompts.softcopy

        it = zip(prompts.natural, prompts.cardinal)
        iprompt = prompts.iprompt.values
        outpaths = np.fromiter((
            Path(outdir, cardinal, f'{natural}.parquet').expanduser()
            for natural, cardinal in it
        ), dtype=object, count=len(prompts))
        index = pd.Index(prompts.natural)
        # noinspection PyTypeChecker
        result: Self = self.enchant({
            'outpath': outpaths,
            'iprompt': iprompt,
        }, index=index)
        return result

    # @Scored
    # def scored(self):
    #     ...

    @classmethod
    def from_directory(
            cls,
            directory: str | Path,
    ) -> Self:
        # files = list(Path(directory).glob('*.parquet'))a
        files = list(
            Path(directory)
            .expanduser()
            .resolve()
            .rglob('*.parquet')
        )
        natural = np.fromiter((
            Path.stem
            for Path in files
        ), dtype=object, count=len(files))
        index = pd.Index(natural, name='natural')
        result = cls({
            'outpath': files,
        }, index=index)
        return result
