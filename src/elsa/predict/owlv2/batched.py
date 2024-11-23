
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
    from elsa.predict.owlv2.magic import OwlV2

from PIL import Image
import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers.utils import TensorType
from transformers.image_transforms import center_to_corners_format
from elsa.predict.owl import batched

def custom_post_process_object_detection(
        outputs, threshold: float = 0.1, target_sizes: Union[TensorType, List[Tuple]] = None
):
    """
    Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
    bottom_right_x, bottom_right_y) format.

    Args:
        outputs ([`OwlViTObjectDetectionOutput`]):
            Raw outputs of the model.
        threshold (`float`, *optional*):
            Score threshold to keep object detection predictions.
        target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
            `(height, width)` of each image in the batch. If unset, predictions will not be resized.
    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
        in the batch as predicted by the model.
    """
    logits, boxes = outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    probs = torch.max(logits, dim=-1)  # TODO: THIS HAS TO CHANGE IN OUR EVALUTION!
    scores = torch.sigmoid(probs.values)
    labels = probs.indices

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})

    return results

class Batched(
    batched.Batched,
):
    outer: OwlV2

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

        if isinstance(files, int):
            loc = elsa.files.file.iunique < files
            files = elsa.files.loc[loc]
        elif files is None:
            files = elsa.files
        else:
            files = elsa.files.loc[files]

        loader = ImageLoader(files, batch_size)

        counter = tqdm.tqdm(total=len(process))
        elsa = process.elsa
        futures = []
        outpath: Path
        empty = pd.DataFrame()
        prompts = process.prompts

        # noinspection PyTypeChecker
        it = zip(
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
                list_score_owlv2 = []
                skip = False

                # infer and concatenate results
                for images, files, pils in loader:
                    captions = [[natural + '.'] * len(images)]
                    inputs = processor(
                        text=captions,
                        # text=[natural],
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
                    # outputs = custom_post_process_object_detection(
                    #     outputs=outputs,
                    #     target_sizes=target_sizes,
                    #     threshold=.1,
                    # )
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
                        score_owlv2 = (
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
                        list_score_owlv2.append(score_owlv2)

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
                score_owlv2 = np.concatenate(list_score_owlv2)
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
                    score_owlv2=score_owlv2,
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
