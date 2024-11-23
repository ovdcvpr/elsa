from __future__ import annotations

import os.path
import os.path
import os.path
from concurrent.futures import ThreadPoolExecutor
from functools import *
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
import tqdm

from elsa.local import local
from elsa.predict.directory import Directory
from elsa.predict.owl.loader import ImageLoader
from elsa.prediction.prediction import Prediction

if False:
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa
    from elsa.predict.owl.magic import Owl


class Batched(
    Directory
):
    outer: Owl

    # noinspection PyTypeHints,PyMethodOverriding
    def __call__(
            self,
            outdir: str = None,
            batch_size: int = None,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            **kwargs,
    ) -> Self:
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
        process: Batched = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        # todo: how do we support super.from params
        # process: Batched = super(self.__class__, self).__from_params__(outdir, prompts, *args, **kwargs)
        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

        elsa: Elsa = process.elsa
        if not force:
            loc = process.exists
            process = process.loc[~loc]
        if not len(process):
            return PROCESS
        # process = process.softcopy

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]
        # files = files.softcopy

        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        elsa = process.elsa
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
        processor = self.outer.processor
        model = self.outer.model
        nskipped = 0

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
                list_score_owl = []
                skip = False

                # infer and concatenate results
                for images, files, pils in loader:
                    captions = [[natural] * len(images)]
                    inputs = processor(
                        text=captions,
                        images=pils,
                        return_tensors='pt'
                    )
                    inputs = {
                        key: value.to('cuda')
                        for key, value in inputs.items()
                    }
                    try:
                        outputs = model(**inputs)
                    except RuntimeError:
                        skip = True
                        break
                    target_sizes = torch.tensor([
                        image.size[::-1]
                        for image in pils
                    ], dtype=torch.float32)
                    outputs = processor.post_process_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=.1,
                    )

                    it = zip(
                        outputs,
                        files.file,
                        files.path,
                        files.ifile,
                    )
                    for output, file, path, ifile in it:
                        boxes = (
                            output['boxes']
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        score_owl = (
                            output['scores']
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        repeat = boxes.shape[0]
                        list_wsen.append(boxes.reshape(-1, 4))
                        list_file.append([file] * repeat)
                        list_path.append([path] * repeat)
                        list_ifile.append([ifile] * repeat)
                        list_score_owl.append(score_owl)

                if skip:
                    nskipped += 1
                    continue
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
                score_owl = np.concatenate(list_score_owl)
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
                    score_owl=score_owl,
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

            msg = (
                f'{nskipped} out of {len(prompts)} prompts skipped '
                f'for being too long for OWL.'
            )
            if nskipped:
                self.logger.warning(msg)
            for future in futures:
                future.result()

            return PROCESS
