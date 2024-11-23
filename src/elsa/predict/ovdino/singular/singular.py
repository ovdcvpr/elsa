from __future__ import annotations
from elsa.resource import  Resource
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
import magicpandas as magic
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from itertools import *

import os
import os.path
import os.path
import os.path
import os.path
import os.path
import os.path
from concurrent.futures import ThreadPoolExecutor
from functools import *
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import tqdm

from elsa.local import local
from elsa.predict.directory import Directory
from elsa.predict.ovdino.batched import batched
from elsa.predict.ovdino.loader import ImageLoader
from elsa.prediction.prediction import Prediction

if False:
    from elsa.predict.ovdino.magic import Magic
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa


class Singular(
    # Resource,
    Directory
):
    outer: Magic

    # noinspection PyTypeHints,PyMethodOverriding
    def __call__(
            self,
            outdir: str = None,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            **kwargs,
    ) -> Self:
        batch_size = 1
        outdir = outdir or os.path.join(os.getcwd(), self.outer.name)
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
        process: Singular = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

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

        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        # noinspection PyTypeChecker
        it = zip(
            # prompts.ilabels.values,
            prompts.ilabels_string.values,
            prompts.natural.values,
            prompts.cardinal.values,
            prompts.catchars.values,
            prompts.labelchars.values,
            process.outpath,
        )

        demo = self.demo
        with ThreadPoolExecutor() as threads:
            submit = partial(threads.submit, pd.DataFrame.to_parquet)
            for (
                    ilabels_string,
                    natural,
                    cardinal,
                    catchars,
                    labelchars,
                    outpath,
            ) in it:
                list_wsen = []
                list_file = []
                list_path = []
                list_ifile = []
                list_score_ovdino = []

                # infer and concatenate results
                for images, files in loader:
                    image = images[0]
                    output = demo.run_on_image(
                        image,
                        category_names=[natural],
                        threshold=0.,
                    )

                    it = zip(
                        files.file,
                        files.path,
                        files.ifile,
                    )
                    for file, path, ifile in it:
                        instances = output['instances']
                        score_ovdino = instances.scores.cpu().numpy()
                        boxes = instances.pred_boxes.tensor.cpu().numpy()
                        repeat = boxes.shape[0]
                        list_wsen.append(boxes.reshape(-1, 4))
                        list_file.append([file] * repeat)
                        list_path.append([path] * repeat)
                        list_ifile.append([ifile] * repeat)
                        list_score_ovdino.append(score_ovdino)

                try:
                    xywh = np.concatenate(list_wsen)
                except ValueError:
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    future = submit(empty, outpath)
                    futures.append(future)
                    counter.update()
                    continue

                # dataframe of other columns
                file = pd.Categorical(np.concatenate(list_file))
                ifile = pd.Categorical(np.concatenate(list_ifile))
                _prompt = pd.Categorical([natural] * len(xywh))
                ilabels_string = (
                    pd.Categorical(
                        [ilabels_string],
                        dtype=prompts.ilabels_string.dtype
                    )
                    .repeat(len(ifile))
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
                    prompt=_prompt,
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
                result = others.assign(natural=natural)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                future = submit(result, outpath)
                futures.append(future)
                counter.update()

            for future in futures:
                future.result()

            return PROCESS

    @property
    def demo(self):
        from elsa.predict.ovdino.singular.predictor import OVDINODemo
        result = OVDINODemo(
            model=self.outer.model,
            sam_predictor=None,
            img_format="BGR",
        )
        return result
