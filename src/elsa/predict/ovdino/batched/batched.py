from __future__ import annotations

from csv import excel

import itertools

import os
import os.path
import os.path
import os.path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import *
from pathlib import Path
from typing import *

import math
import numpy as np
import pandas as pd
import tqdm

from elsa.local import local
from elsa.predict.directory import Directory
from elsa.predict.ovdino.loader import ImageLoader
from elsa.predict.ovdino.singular.singular import Singular
from elsa.prediction.prediction import Prediction

if False:
    from elsa.predict.ovdino.magic import Magic
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa

"""

"""


class Batched(
    Singular,
):
    outer: Magic

    # noinspection PyTypeHints,PyMethodOverriding
    def __call__(
            self,
            outdir: str = None,
            batch_size: int = None,
            # force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            **kwargs,
    ) -> Self:
        force = True
        outdir: Any = outdir or os.path.join(os.getcwd(), self.outer.__name__)
        outdir: Path = (
            Path(outdir)
            .expanduser()
            .resolve()
        )
        if prompts is None:
            prompts = self.elsa.prompts
        elif isinstance(prompts, int):
            prompts = self.elsa.prompts.iloc[:prompts]
        else:
            prompts = self.elsa.prompts.loc[prompts]
        if synonyms:
            prompts = prompts.limit_synonyms(n=synonyms)
            assert (
                       prompts
                       .groupby('ilabels')
                       .size()
                       .max()
                   ) <= synonyms
        process: Batched = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

        if batch_size != 1:
            msg = (
                f'Currently OVDINO is not supported for batches '
                f'that are not of size 1. '
            )
            raise NotImplementedError(msg)

        elsa: Elsa = process.elsa
        if not force:
            loc = process.exists
            process = process.loc[~loc]
        if not len(process):
            return PROCESS

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]

        loader = ImageLoader(files, batch_size=1)

        counter = tqdm.tqdm(total=len(files))
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        # noinspection PyTypeChecker
        n = math.ceil(len(prompts) / batch_size)
        igroup = (
            np.arange(n)
            .repeat(batch_size)
            [:len(prompts)]
        )
        prompts = prompts.assign(ilabels_string=prompts.ilabels_string.values)
        # noinspection PyTypeChecker
        groups: Iterable[Prompts] = prompts.groupby(igroup)
        groups = list(groups)
        tempdir = Path(tempfile.mkdtemp())

        with ThreadPoolExecutor() as threads:
            submit = partial(threads.submit, pd.DataFrame.to_parquet)
            for images, files in loader:
                image = images[0]
                list_wsen = []
                list_score_ovdino = []
                list_natural = []
                list_ilabels_string = []
                n = 0
                outpath = tempdir / f'{files.ifile.values[0]}.parquet'


                # the last batch might have a different size;
                # this may cause an error for the model
                most = zip(groups[:-1], itertools.repeat(self.demo))
                last = zip(groups[-1:], itertools.repeat(self.demo))
                it = itertools.chain(most, last)

                for (i, group), demo in it:
                    ilabels_string = group.ilabels_string.values
                    natural = group.natural.values
                    output = demo.run_on_image(
                        image,
                        category_names=natural,
                        threshold=0.,
                    )
                    instances = output['instances']
                    score_ovdino = instances.scores.cpu().numpy()
                    boxes = instances.pred_boxes.tensor.cpu().numpy()
                    repeat = boxes.shape[0]
                    iloc = instances.pred_classes.cpu().numpy()
                    n += repeat

                    list_wsen.append(boxes.reshape(-1, 4))
                    list_natural.append(natural[iloc])
                    list_score_ovdino.append(score_ovdino)
                    list_ilabels_string.append(ilabels_string[iloc])

                ifile = files.ifile.values.repeat(n)
                file = files.file.values.repeat(n)
                prompt = np.concatenate(list_natural)
                prompt = pd.Categorical(prompt)

                path = tempdir / f'{files.ifile.values[0]}.parquet'
                try:
                    xywh = np.concatenate(list_wsen)
                except ValueError:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    future = submit(empty, outpath)
                    futures.append(future)
                    counter.update()
                    continue

                # dataframe of other columns
                file = pd.Categorical(file)
                ifile = pd.Categorical(ifile)
                ilabels_string = pd.Categorical(
                    np.concatenate(list_ilabels_string),
                    categories=prompts.ilabels_string.cat.categories,
                )
                score_ovdino = np.concatenate(list_score_ovdino)
                wsen = np.concatenate(list_wsen)

                w = wsen[:, 0]
                s = wsen[:, 1]
                e = wsen[:, 2]
                n = wsen[:, 3]

                others = pd.DataFrame(dict(
                    w=w,
                    s=s,
                    e=e,
                    n=n,
                    file=file,
                    prompt=prompt,
                    ifile=ifile,
                    ilabels_string=ilabels_string,
                    score_ovdino=score_ovdino,
                ))
                index = others.columns
                empty = [''] * len(index)
                arrays = index, empty, empty, empty
                names = Prediction.levels
                columns = pd.MultiIndex.from_arrays(arrays, names=names)
                others.columns = columns

                # concat and submit

                result = others
                # result = others.assign(natural=natural)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                future = submit(result, outpath)
                futures.append(future)
                counter.update()

            for future in futures:
                future.result()

            naturals = prompts.natural.values
            outdir.mkdir(parents=True, exist_ok=True)

            def filter_parquet_for_prompt(parquet_file, prompt):
                df = pd.read_parquet(parquet_file)
                return df[df['prompt'] == prompt]

            def write_parquet_file(df, outdir, prompt):
                if df:
                    result_df = pd.concat(df, ignore_index=True)
                    result_df.to_parquet(outdir / f"{prompt}.parquet")

            with ThreadPoolExecutor() as executor:
                futures = []
                for prompt in naturals:
                    prompt_dfs = [
                        executor.submit(filter_parquet_for_prompt, parquet_file, prompt)
                        for parquet_file in tempdir.glob("*.parquet")
                    ]
                    filtered_dfs = [
                        future.result()
                        for future in prompt_dfs
                        if not future.result().empty
                    ]
                    future = executor.submit(
                        write_parquet_file,
                        filtered_dfs,
                        outdir,
                        prompt,
                    )
                    futures.append(future)

                for future in futures:
                    future.result()

            return PROCESS
