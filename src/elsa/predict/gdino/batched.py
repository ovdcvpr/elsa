from __future__ import annotations

import os.path
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import *
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
import tqdm
from pandas import Series, DataFrame, MultiIndex

from elsa.local import local
from elsa.predict import util
from elsa.predict.directory import Directory
from elsa.predict.gdino import loader
from elsa.predict.gdino.loader import ImageLoader
from elsa.prediction.prediction import Prediction
from elsa.predict.util import resolve_duplicates, replace_column_names

if False:
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa
    from elsa.files.files import Files
    from elsa.predict.gdino.gdino import GDino
    from elsa.predict.gdino.model import Result


class Columns:

    @classmethod
    def from_string(
            cls,
            columns: str = 'normx normy normxminidth normheight file path prompt'
    ) -> MultiIndex:
        if isinstance(columns, str):
            columns = columns.split()
        empty = [''] * len(columns)
        arrays = columns, empty, empty, empty
        result = pd.MultiIndex.from_arrays(arrays, names=Prediction.levels)
        return result

    @classmethod
    def from_confidence(
            cls,
            natural: str,
            elsa: Elsa,
            offset_mapping,
            catchars,
            labelchars,
    ):
        token = np.fromiter((
            natural[ifirst:ilast]
            for ifirst, ilast in offset_mapping
        ), dtype=object, count=offset_mapping.shape[0])
        catchars = util.parse(catchars, offset_mapping, natural)
        labelchars = util.parse(labelchars, offset_mapping, natural)
        labels = (
            elsa.labels
            .char2label
            .loc[labelchars]
            .values
        )
        cat2char = elsa.cat2char
        char2cat = Series(cat2char.index, index=cat2char.values)
        cat = char2cat.loc[catchars].values
        ifirst = offset_mapping[:, 0].astype(str)
        # arrays = cat, labels, token, ifirst
        # arrays = ifirst, token, labels, cat
        arrays = token, ifirst, labels, cat
        columns = pd.MultiIndex.from_arrays(arrays, names=Prediction.levels)
        return columns



class Batched(
    Directory
):
    outer: GDino

    # noinspection PyTypeHints,PyMethodOverriding
    def __call__(
            self,
            outdir: str = None,
            batch_size: int = None,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            config=None,
            checkpoint=None,
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
        config = config or local.predict.gdino.config
        checkpoint = checkpoint or local.predict.gdino.checkpoint
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

        if not os.path.exists(checkpoint):
            checkpoint_url = (
                'https://github.com/longzw1997/Open-GroundingDino/'
                'releases/download/v0.1.0/gdinot-coco-ft.pth'
            )
            msg = f'Checkpoint file not found at {checkpoint}. '
            msg += f'Downloading from {checkpoint_url}, please wait...'
            self.logger.info(msg)
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint)

        loader = ImageLoader(files, batch_size)

        # else:
        #     loader = ImageLoader(elsa.files.loc[files], batch_size)
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
        model = self.outer.model(config, checkpoint)

        with ThreadPoolExecutor() as threads:
            submit = partial(threads.submit, pd.DataFrame.to_parquet)
            for (
                    # ilabels,
                    ilabels_string,
                    natural,
                    cardinal,
                    catchars,
                    labelchars,
                    outpath,
            ) in it:
                captions = [natural + '.']  # gdino requires .
                offset_mapping = None
                list_confidence = []
                list_xywh = []
                list_file = []
                list_path = []
                list_ifile = []

                # infer and concatenate results
                for images, files in loader:
                    with torch.no_grad():
                        outputs: Result = model(images, captions=captions * images.shape[0])  # replicating the caption batch_size times
                    offset_mapping = outputs.offset_mapping[outputs.icol]
                    # todo: check this
                    list_confidence.append(outputs.confidence.reshape(-1, len(outputs.icol)))

                    repeat = outputs.xywh.shape[1]
                    list_xywh.append(outputs.xywh.reshape(-1, 4))
                    list_file.append(files.file.values.repeat(repeat))
                    list_path.append(files.path.values.repeat(repeat))
                    list_ifile.append(files.ifile.values.repeat(repeat))

                # if empty just write an empty frame to file
                try:
                    xywh = np.concatenate(list_xywh)
                except ValueError:
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    future = submit(empty, outpath)
                    futures.append(future)
                    counter.update()
                    continue

                # confidence dataframe
                data = np.concatenate(list_confidence, axis=0)
                columns = Columns.from_confidence(
                    natural=natural,
                    elsa=elsa,
                    offset_mapping=offset_mapping,
                    catchars=catchars,
                    labelchars=labelchars,
                )
                confidence = DataFrame(data, columns=columns)

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

                normx = xywh[:, 0]
                normy = xywh[:, 1]
                normxminidth = xywh[:, 2]
                normheight = xywh[:, 3]
                normxmin = np.maximum(0, normx - normxminidth / 2)
                normxmax = np.minimum(1, normx + normxminidth / 2)
                normymin = np.maximum(0, normy - normheight / 2)
                normymax = np.minimum(1, normy + normheight / 2)

                others = pd.DataFrame(dict(
                    normxmin=normxmin,
                    normxmax=normxmax,
                    normymin=normymin,
                    normymax=normymax,
                    file=file,
                    prompt=_prompt,
                    ifile=ifile,
                    ilabels_string=ilabels_string,
                ))
                index = others.columns
                empty = [''] * len(index)
                arrays = index, empty, empty, empty
                names = Prediction.levels
                columns = pd.MultiIndex.from_arrays(arrays, names=names)
                others.columns = columns

                # concat and submit
                result = pd.concat([confidence, others], axis=1)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                future = submit(result, outpath)
                futures.append(future)
                counter.update()

            for future in futures:
                future.result()

            return PROCESS

    def reconstruct(
            self,
            outdir: str = None,
            batch_size: int = None,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            config=None,
            checkpoint=None,
    ) -> Self:
        force = True,
        files: int | Collection[bool] = 1
        resolve_duplicates(outdir, logger=self.logger)
        rglob = (
            Path(outdir)
            .expanduser()
            .resolve()
            .rglob('*.parquet')
        )
        natural2path: dict[str, Path] = {
            path.stem: path
            for path in rglob
        }

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
        config = config or local.predict.gdino.config
        checkpoint = checkpoint or local.predict.gdino.checkpoint
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

        if not os.path.exists(checkpoint):
            checkpoint_url = (
                'https://github.com/longzw1997/Open-GroundingDino/'
                'releases/download/v0.1.0/gdinot-coco-ft.pth'
            )
            msg = f'Checkpoint file not found at {checkpoint}. '
            msg += f'Downloading from {checkpoint_url}, please wait...'
            self.logger.info(msg)
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint)

        loader = ImageLoader(files, batch_size)

        # else:
        #     loader = ImageLoader(elsa.files.loc[files], batch_size)
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
        model = self.outer.model(config, checkpoint)
        at_least_one = False

        with ThreadPoolExecutor() as threads:
            submit = partial(threads.submit, pd.DataFrame.to_parquet)
            for (
                    # ilabels,
                    ilabels_string,
                    natural,
                    cardinal,
                    catchars,
                    labelchars,
                    outpath,
            ) in it:
                at_least_one = True
                if natural not in natural2path:
                    counter.update()
                    continue
                path = natural2path[natural]
                natural2path.pop(natural)
                captions = [natural + '.']  # gdino requires .
                offset_mapping = None
                list_confidence = []
                list_xywh = [[]]
                list_file = [[]]
                list_path = [[]]
                list_ifile = [[]]

                # infer and concatenate results
                for images, files in loader:
                    with torch.no_grad():
                        outputs: Result = model(images, captions=captions * images.shape[0])  # replicating the caption batch_size times
                    offset_mapping = outputs.offset_mapping[outputs.icol]

                # if empty just write an empty frame to file
                try:
                    xywh = np.concatenate(list_xywh)
                except ValueError:
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    future = submit(empty, outpath)
                    futures.append(future)
                    counter.update()
                    continue

                # confidence dataframe

                data = np.empty((0, len(offset_mapping)))
                columns = Columns.from_confidence(
                    natural=natural,
                    elsa=elsa,
                    offset_mapping=offset_mapping,
                    catchars=catchars,
                    labelchars=labelchars,
                )
                confidence = DataFrame(data, columns=columns)

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

                xywh = np.empty((0, 4))
                normx = xywh[:, 0]
                normy = xywh[:, 1]
                normxminidth = xywh[:, 2]
                normheight = xywh[:, 3]
                normxmin = np.maximum(0, normx - normxminidth / 2)
                normxmax = np.minimum(1, normx + normxminidth / 2)
                normymin = np.maximum(0, normy - normheight / 2)
                normymax = np.minimum(1, normy + normheight / 2)

                others = pd.DataFrame(dict(
                    normxmin=normxmin,
                    normxmax=normxmax,
                    normymin=normymin,
                    normymax=normymax,
                    file=file,
                    prompt=_prompt,
                    ifile=ifile,
                    ilabels_string=ilabels_string,
                ))
                index = others.columns
                empty = [''] * len(index)
                arrays = index, empty, empty, empty
                names = Prediction.levels
                columns = pd.MultiIndex.from_arrays(arrays, names=names)
                others.columns = columns

                # concat and submit
                result = pd.concat([confidence, others], axis=1)

                future = threads.submit(replace_column_names, path, result.columns)
                futures.append(future)
                counter.update()

            assert at_least_one, 'No files were found'
            for future in futures:
                future.result()
            if natural2path:
                msg = (
                    f"Found {len(natural2path)} prediction files not "
                    f"represented by the current prompts"
                )
                self.logger.warning(msg)
                for path in natural2path.values():
                    self.logger.warning(f"Deleting {path}")
                    path.unlink(missing_ok=True)
            return