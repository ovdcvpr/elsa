from __future__ import annotations
from mdetr.models import build_model
import torchvision.transforms as T

import os.path
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
# from elsa.predict.gdino import loader
from elsa.predict.mdetr.loader import ImageLoader
from elsa.predict.mdetr.parser import get_args_parser
from elsa.predict.util import resolve_duplicates, replace_column_names
from elsa.prediction.prediction import Prediction

if False:
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa
    from elsa.predict.mdetr.model import Result, MDETR


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


class Batched(Directory):
    def __call__(
            self,
            outdir: str = None,
            batch_size: int = None,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            _model=None,
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        from elsa.predict.mdetr.model import MDETR
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
                    <= synonyms
            )
        # prompts = prompts.softcopy

        process: Batched = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        # todo: how do we support super.from params
        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

        elsa: Elsa = process.elsa
        if not force:
            loc = process.exists
            process = process.loc[~loc]
        if not len(process):
            return PROCESS
        process = process.copy()

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]
        files = files.copy()
        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        elsa = process.elsa
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        it = zip(
            # prompts.ilabels.values,
            prompts.ilabels_string.values,
            prompts.natural.values,
            prompts.cardinal.values,
            prompts.catchars.values,
            prompts.labelchars.values,
            process.outpath,
        )
        if _model is not None:
            model = _model
        else:
            model = MDETR.from_elsa(**kwargs)
        torch.set_grad_enabled(False)
        # model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
        # model = model.cuda()
        # model.eval()

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
                captions = [natural]
                caption = natural
                offset_mapping = None
                list_confidence = []
                list_xywh = []
                list_file = []
                list_path = []
                list_ifile = []
                list_score_mdetr = []

                # infer and concatenate results
                for images, files in loader:
                    memory_cache = model(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=True,
                    )
                    outputs: Result = model(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=False,
                        memory_cache=memory_cache,
                    )

                    ncolumns = memory_cache['tokenized'].input_ids.shape[1] - 1
                    columns = list(range(1, ncolumns))
                    queries = outputs["proj_queries"]
                    tokens = outputs["proj_tokens"].transpose(-1, -2)
                    logits = torch.matmul(queries, tokens) / temperature
                    scores = (
                        torch.sigmoid(logits[:, :, 1:-1])
                        .cpu()
                        .numpy()
                    )
                    nbatches, nbatch, ntokens = scores.shape
                    scores = scores.reshape((nbatches * nbatch, ntokens))
                    ntokens = scores.shape[1]

                    xywh = (
                        outputs['pred_boxes']
                        .cpu()
                        .numpy()
                        .reshape(-1, 4)
                    )

                    # print(f"{outputs['pred_logits'].shape=}")

                    assert len(xywh) == len(scores)
                    # see https://colab.research.google.com/github/ashkamath/mdetr/blob/colab/notebooks/MDETR_demo.ipynb#scrollTo=vRw-Bhf8QTAh
                    score_mdetr = (
                        outputs['pred_logits']
                        .softmax(-1)
                        [:, :, -1]
                        .cpu()
                        .__rsub__(1)
                        .ravel()
                        .numpy()
                    )
                    # score_mdetr = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
                    # loc = score_mdetr > .7
                    # loc.sum()

                    repeat = outputs.xywh.shape[1]
                    # repeat = xywh.shape[0]
                    list_confidence.append(scores)
                    list_xywh.append(xywh)
                    list_file.append(files.file.values.repeat(repeat))
                    list_path.append(files.path.values.repeat(repeat))
                    list_ifile.append(files.ifile.values.repeat(repeat))
                    list_score_mdetr.append(score_mdetr)

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
                tokenized = memory_cache['tokenized']
                offset_mapping = [
                    list(tokenized.token_to_chars(0, i))
                    for i in columns
                ]
                offset_mapping = np.array(offset_mapping)

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
                score_mdetr = np.concatenate(list_score_mdetr)

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
                    score_mdetr=score_mdetr,
                    ntokens=ntokens,
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


    # noinspection PyTypeHints,PyMethodOverriding
    def test(
            self,
            outdir: str = None,
            batch_size: int = None,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            _model=None,
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        from elsa.predict.mdetr.model import MDETR
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
                    <= synonyms
            )
        # prompts = prompts.softcopy

        process: Batched = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        # todo: how do we support super.from params
        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

        elsa: Elsa = process.elsa
        if not force:
            loc = process.exists
            process = process.loc[~loc]
        if not len(process):
            return PROCESS
        process = process.copy()

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]
        files = files.copy()
        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        elsa = process.elsa
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        if _model is not None:
            model = _model
        else:
            model = MDETR.from_elsa(**kwargs)
        torch.set_grad_enabled(False)

        local = self._local(
            outdir='/tmp/mdetr',
            batch_size=1,
            load='/home/redacted/Downloads/pretrained_EB5_checkpoint.pth',
            backbone='timm_tf_efficientnet_b5_ns',
        )
        # for (local_name, local_param), (hub_name, hub_param) in zip(local_model.named_parameters(), hub_model.named_parameters()):
        #     assert torch.allclose(local_param, hub_param, atol=1e-6), f"Parameter mismatch in layer {local_name}"
        it = zip(local.named_parameters(), model.named_parameters())
        mismatches = [
            (ln, hn) for (ln, lp), (hn, hp) in it if not torch.allclose(lp, hp, atol=1e-6)
        ]
        if mismatches:
            raise ValueError(f"Parameter mismatches in layers: {', '.join(f'{ln} vs {hn}' for ln, hn in mismatches)}")

        it = zip(local.state_dict().items(), model.state_dict().items())
        mismatches = [
            (ln, hn) for (ln, lp), (hn, hp) in it if not torch.allclose(lp, hp, atol=1e-6)
        ]
        if mismatches:
            raise ValueError(f"Parameter mismatches in layers: {', '.join(f'{ln} vs {hn}' for ln, hn in mismatches)}")

        it = zip(
            # prompts.ilabels.values,
            prompts.ilabels_string.values,
            prompts.natural.values,
            prompts.cardinal.values,
            prompts.catchars.values,
            prompts.labelchars.values,
            process.outpath,
        )
        # model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
        # model = model.cuda()
        # model.eval()

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
                captions = [natural]
                caption = natural
                offset_mapping = None
                list_confidence = []
                list_xywh = []
                list_file = []
                list_path = []
                list_ifile = []
                list_score_mdetr = []

                # infer and concatenate results
                for images, files in loader:
                    memory_cache = model(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=True,
                    )
                    outputs: Result = model(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=False,
                        memory_cache=memory_cache,
                    )

                    local_memory = local(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=True,
                    )
                    local_outputs: Result = local(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=False,
                        memory_cache=local_memory,
                    )
                    # mismatches = [
                    #     key for key in outputs.keys()
                    #     if not torch.allclose(outputs[key], local_outputs[key], atol=1e-6)
                    # ]
                    mismatches = []
                    for key in outputs.keys():
                        try:
                            if not torch.allclose(outputs[key], local_outputs[key], atol=1e-6):
                                mismatches.append(key)
                        except TypeError:
                            ...
                    if mismatches:
                        raise ValueError(f"Mismatched output keys: {', '.join(mismatches)}")

                    ncolumns = memory_cache['tokenized'].input_ids.shape[1] - 1
                    columns = list(range(1, ncolumns))
                    queries = outputs["proj_queries"]
                    tokens = outputs["proj_tokens"].transpose(-1, -2)
                    logits = torch.matmul(queries, tokens) / temperature
                    scores = (
                        torch.sigmoid(logits[:, :, 1:-1])
                        .cpu()
                        .numpy()
                    )
                    nbatches, nbatch, ntokens = scores.shape
                    scores = scores.reshape((nbatches * nbatch, ntokens))

                    xywh = (
                        outputs['pred_boxes']
                        .cpu()
                        .numpy()
                        .reshape(-1, 4)
                    )

                    print(f"{outputs['pred_logits'].shape=}")

                    assert len(xywh) == len(scores)
                    # see https://colab.research.google.com/github/ashkamath/mdetr/blob/colab/notebooks/MDETR_demo.ipynb#scrollTo=vRw-Bhf8QTAh
                    score_mdetr = (
                        outputs['pred_logits']
                        .softmax(-1)
                        [:, :, -1]
                        .cpu()
                        .__rsub__(1)
                        .ravel()
                        .numpy()
                    )
                    # score_mdetr = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
                    # loc = score_mdetr > .7
                    # loc.sum()

                    # repeat = outputs.xywh.shape[1]
                    repeat = xywh.shape[0]
                    list_confidence.append(scores)
                    list_xywh.append(xywh)
                    list_file.append(files.file.values.repeat(repeat))
                    list_path.append(files.path.values.repeat(repeat))
                    list_ifile.append(files.ifile.values.repeat(repeat))
                    list_score_mdetr.append(score_mdetr)

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
                tokenized = memory_cache['tokenized']
                offset_mapping = [
                    list(tokenized.token_to_chars(0, i))
                    for i in columns
                ]
                offset_mapping = np.array(offset_mapping)

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
                score_mdetr = np.concatenate(list_score_mdetr)

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
                    score_mdetr=score_mdetr,
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

    def local(
            self,
            *args,
            load: str,
            outdir: str = None,
            text_encoder_type: str = 'roberta-base',
            backbone: str = 'timm_tf_efficientnet_b3_ns',
            batch_size: int = 1,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            device='cuda',
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        """Instantiate the model locally instead of torch.hub.load"""
        parse = [
            f'--{arg}'
            for arg in args
        ]
        for key, value in kwargs.items():
            parse.append(f'--{key}')
            parse.append(str(value))
        parse.append('--device')
        parse.append(device)
        parse.append('--load')
        parse.append(load)
        parse.append('--text_encoder_type')
        parse.append(text_encoder_type)
        parse.append('--backbone')
        parse.append(backbone)
        # parse.append('--output_dir')
        # parse.append(outdir)
        parse.append('--batch_size')
        parse.append(str(batch_size))

        parsed = get_args_parser().parse_args(parse)

        model, criterion, contrastive_criterion, postprocessors, weight_dict = build_model(parsed)
        model: MDETR

        # Load the model weights (checkpoint)
        checkpoint = torch.load(parsed.load, map_location=parsed.device)
        model.load_state_dict(checkpoint['model'])
        model.to(parsed.device)

        return self(
            outdir=outdir,
            batch_size=batch_size,
            temperature=temperature,
            force=force,
            synonyms=synonyms,
            prompts=prompts,
            files=files,
            _model=model,
            **kwargs,
        )

    def _test(
            self,
            file: str = None,
            prompt: str = None,
    ):
        """
        Import plot_results, plotting a visualization using their
        provided example. Then, run __call__ for that same file and
        prompt, setting a breakpoint once the scores are returned.
        Confirm that the top N values are the same as the N boxes
        shown in the visualization.
        """

    def reconstruct(
            self,
            outdir: str = None,
            batch_size: int = None,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = 1,
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        """
        😳
        """
        force = True,
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

        from elsa.predict.mdetr.model import MDETR
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
                    <= synonyms
            )

        process: Batched = (
            Directory
            .__from_method__
            .__get__(self, Directory)
            (outdir, prompts)
        )

        # todo: how do we support super.from params
        PROCESS = process
        batch_size = batch_size or local.predict.gdino.batch_size

        elsa: Elsa = process.elsa
        if not force:
            loc = process.exists
            process = process.loc[~loc]
        if not len(process):
            return PROCESS
        process = process.copy()

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]
        files = files.copy()

        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        elsa = process.elsa
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        it = zip(
            # prompts.ilabels.values,
            prompts.ilabels_string.values,
            prompts.natural.values,
            prompts.cardinal.values,
            prompts.catchars.values,
            prompts.labelchars.values,
            process.outpath,
        )
        model = MDETR.from_elsa(**kwargs)
        torch.set_grad_enabled(False)
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
                captions = [natural]
                caption = natural
                offset_mapping = None
                # list_confidence = [[]] * len(offset_mapping)
                list_xywh = [[]]
                list_file = [[]]
                list_path = [[]]
                list_ifile = [[]]

                # infer and concatenate results
                for images, files in loader:
                    memory_cache = model(
                        images,
                        captions=captions * images.shape[0],
                        encode_and_save=True,
                    )
                    ncolumns = memory_cache['tokenized'].input_ids.shape[1] - 1
                    columns = list(range(1, ncolumns))

                try:
                    xywh = np.concatenate(list_xywh)
                except ValueError:
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    future = submit(empty, outpath)
                    futures.append(future)
                    counter.update()
                    continue

                # confidence dataframe
                tokenized = memory_cache['tokenized']
                offset_mapping = [
                    list(tokenized.token_to_chars(0, i))
                    for i in columns
                ]
                offset_mapping = np.array(offset_mapping)

                # data = np.concatenate(list_confidence, axis=0)
                columns = Columns.from_confidence(
                    natural=natural,
                    elsa=elsa,
                    offset_mapping=offset_mapping,
                    catchars=catchars,
                    labelchars=labelchars,
                )
                data = np.empty((0, len(columns)))
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
                # path = natural2path[natural]

                future = threads.submit(replace_column_names, path, result.columns)
                futures.append(future)
                # natural2path.pop(natural)
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

    def _local(
            self,
            *args,
            load: str,
            outdir: str = None,
            text_encoder_type: str = 'roberta-base',
            backbone: str = 'timm_tf_efficientnet_b3_ns',
            batch_size: int = 1,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            device='cuda',
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> MDETR:
        """Instantiate the model locally instead of torch.hub.load"""
        parse = [
            f'--{arg}'
            for arg in args
        ]
        for key, value in kwargs.items():
            parse.append(f'--{key}')
            parse.append(str(value))
        parse.append('--device')
        parse.append(device)
        parse.append('--load')
        parse.append(load)
        parse.append('--text_encoder_type')
        parse.append(text_encoder_type)
        parse.append('--backbone')
        parse.append(backbone)
        # parse.append('--output_dir')
        # parse.append(outdir)
        parse.append('--batch_size')
        parse.append(str(batch_size))

        parsed = get_args_parser().parse_args(parse)

        model, criterion, contrastive_criterion, postprocessors, weight_dict = build_model(parsed)
        model: MDETR

        # Load the model weights (checkpoint)
        checkpoint = torch.load(parsed.load, map_location=parsed.device)
        model.load_state_dict(checkpoint['model'])
        model.to(parsed.device)

        return model

    def original(
            self,
            *args,
            load: str,
            outdir: str = None,
            text_encoder_type: str = 'roberta-base',
            backbone: str = 'timm_tf_efficientnet_b3_ns',
            batch_size: int = 1,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            device='cuda',
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        """Instantiate the model locally instead of torch.hub.load"""
        parse = [
            f'--{arg}'
            for arg in args
        ]
        for key, value in kwargs.items():
            parse.append(f'--{key}')
            parse.append(str(value))
        parse.append('--device')
        parse.append(device)
        parse.append('--load')
        parse.append(load)
        parse.append('--text_encoder_type')
        parse.append(text_encoder_type)
        parse.append('--backbone')
        parse.append(backbone)
        # parse.append('--output_dir')
        # parse.append(outdir)
        parse.append('--batch_size')
        parse.append(str(batch_size))

        # parsed = get_args_parser().parse_args(parse)
        from main import get_args_parser
        parsed = get_args_parser().parse_args(parse)

        model, criterion, contrastive_criterion, postprocessors, weight_dict = build_model(parsed)
        model: MDETR

        # Load the model weights (checkpoint)
        checkpoint = torch.load(parsed.load, map_location=parsed.device)
        model.load_state_dict(checkpoint['model'])
        model.to(parsed.device)

        return self(
            outdir=outdir,
            batch_size=batch_size,
            temperature=temperature,
            force=force,
            synonyms=synonyms,
            prompts=prompts,
            files=files,
            _model=model,
            **kwargs,
        )

    def _original(
            self,
            *args,
            load: str,
            outdir: str = None,
            text_encoder_type: str = 'roberta-base',
            backbone: str = 'timm_tf_efficientnet_b3_ns',
            batch_size: int = 1,
            temperature=.1,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            device='cuda',
            # kwargs go to Elsa.from_kwargs which then goes to torch.hub.load
            **kwargs,
    ) -> Self:
        """Instantiate the model locally instead of torch.hub.load"""
        parse = [
            f'--{arg}'
            for arg in args
        ]
        for key, value in kwargs.items():
            parse.append(f'--{key}')
            parse.append(str(value))
        parse.append('--device')
        parse.append(device)
        parse.append('--load')
        parse.append(load)
        parse.append('--text_encoder_type')
        parse.append(text_encoder_type)
        parse.append('--backbone')
        parse.append(backbone)
        # parse.append('--output_dir')
        # parse.append(outdir)
        parse.append('--batch_size')
        parse.append(str(batch_size))

        parsed = get_args_parser().parse_args(parse)

        model, criterion, contrastive_criterion, postprocessors, weight_dict = build_model(parsed)
        model: MDETR

        # Load the model weights (checkpoint)
        checkpoint = torch.load(parsed.load, map_location=parsed.device)
        model.load_state_dict(checkpoint['model'])
        model.to(parsed.device)
        return model
