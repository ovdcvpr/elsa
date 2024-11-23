from __future__ import annotations
from elsa.prediction.prediction import Prediction
import requests

import os
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
from torch.nn import functional as F

from elsa.local import local
from elsa.predict.detic.errors import DeticImportError, DetectronImportError
from elsa.predict.detic.loader import ImageLoader
from elsa.predict.directory import Directory
from elsa.predict.gdino import batched

if False:
    from elsa.predict.detic.magic import Detic
if False:
    from elsa.annotation.prompts import Prompts
    from elsa import Elsa
    from elsa.predict.detic.magic import Magic as Detic

class Batched(
    batched.Batched,
):
    outer: Detic

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
        try:
            from detectron2.data import MetadataCatalog
        except ImportError as e:
            raise DetectronImportError() from e
        try:
            from detic.modeling.text.text_encoder import build_text_encoder
        except ModuleNotFoundError:
            ...
        try:
            from detic.modeling.text.text_encoder import build_text_encoder
        except ImportError as e:
            raise DeticImportError() from e
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

        _ = self.outer.weights

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
        cwd = os.getcwd()
        # os.chdir(__file__)
        path = os.path.join( __file__, '../' )
        path = os.path.abspath(path)
        os.chdir(path)

        # path = "Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"
        # path = os.path.abspath(path)
        # if not os.path.exists(path):
        #     msg = f'Downloading weights to {path}'
        #     self.logger.info(msg)
        #     # URL to download the file
        #     url = "https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"
        #
        #     # Download the file and save it to the specified path
        #     response = requests.get(url, stream=True)
        #     response.raise_for_status()  # Check for download errors
        #
        #     with open(path, 'wb') as f:
        #         for chunk in response.iter_content(chunk_size=8192):
        #             f.write(chunk)

        path = "Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"
        path = os.path.abspath(path)

        if not os.path.exists(path):
            msg = f'Downloading weights to {path}'
            self.logger.info(msg)
            url = "https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"

            try:
                # Start the download with streaming
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for any HTTP errors

                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info("Download completed successfully.")

            except requests.RequestException as e:
                self.logger.error(f"Error downloading weights: {e}")
                if os.path.exists(path):
                    os.remove(path)  # Remove incomplete file if download fails
                raise

        predictor = self.outer.predictor
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
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
                list_score_detic = []

                # texts = [prompt + x for x in vocabulary]
                # zs_weight = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
                zs_weight = (
                    text_encoder([natural])
                    .detach()
                    .permute(1, 0)
                    .contiguous()
                    .cpu()
                )
                metadata = MetadataCatalog.get(natural)
                metadata.thing_classes = [natural]
                predictor.model.roi_heads.num_classes = 1
                zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], dim=1)  # D x (C + 1)
                if predictor.model.roi_heads.box_predictor.cls_score.norm_weight:
                    zs_weight = F.normalize(zs_weight, p=2, dim=0)
                zs_weight = zs_weight.to(predictor.model.device)
                del predictor.model.roi_heads.box_predictor.cls_score.zs_weight
                predictor.model.roi_heads.box_predictor.cls_score.zs_weight = zs_weight

                # infer and concatenate results
                for images, files in loader:
                    outputs = predictor(images)

                    it = zip(
                        outputs,
                        files.file,
                        files.path,
                        files.ifile,
                    )
                    for output, file, path, ifile in it:
                        instances = output['instances']
                        score_detic = instances.scores.cpu().numpy()
                        boxes = instances.pred_boxes.tensor.cpu().numpy()
                        repeat = boxes.shape[0]
                        list_wsen.append(boxes.reshape(-1, 4))
                        list_file.append([file] * repeat)
                        list_path.append([path] * repeat)
                        list_ifile.append([ifile] * repeat)
                        list_score_detic.append(score_detic)

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
                score_detic = np.concatenate(list_score_detic)
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
                    score_detic=score_detic,
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

            os.chdir(cwd)
            return PROCESS
        ...

