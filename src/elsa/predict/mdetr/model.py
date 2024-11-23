from __future__ import annotations
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

import collections
from functools import *

import numpy as np
import torch

from mdetr.models import mdetr


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

    # @dataclass
    # class Result:
    #     pred_logits: torch.Tensor
    #     pred_boxes: torch.Tensor
    #     input_ids: torch.Tensor
    offset_mapping: torch.Tensor


#     memory_cache: torch.Tensor

class Result(collections.UserDict):

    @property
    def pred_logits(self) -> torch.Tensor:
        return self["pred_logits"]

    @property
    def pred_boxes(self) -> torch.Tensor:
        return self["pred_boxes"]

    @property
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]

    @property
    def proj_tokens(self):
        return self["proj_tokens"]

    @property
    def proj_queries(self):
        return self["proj_queries"]



    # @property
    # def offset_mapping(self) -> torch.Tensor:
    #     return self["offset_mapping"]

    # def __post_init__(self):
    #     offset_mapping = (
    #         self.offset_mapping
    #         .squeeze()
    #         .cpu()
    #         # .detach()
    #         .numpy()
    #     )
    #     if len(offset_mapping.shape) == 3:
    #         offset_mapping = offset_mapping[0]
    #     self.offset_mapping = offset_mapping

    @cached_property
    def icol(self):
        offset_mapping = self.offset_mapping
        icol = np.arange(len(offset_mapping))
        loc = icol > 0
        loc &= icol < (len(offset_mapping) - 2)
        icol = icol[loc]
        return icol

    @cached_property
    def probas(self):
        result = (
            self.pred_logits
            .softmax(-1)
            [0, :, :-1]
            .cpu()
            .__rsub__(1)
        )
        return result

    @cached_property
    def keep(self):
        return (
            self.probas
            .gt(.7)
            .cpu()
        )

    @cached_property
    def positive_tokens(self):
        # keep = (probas > 0.7).cpu()
        # positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
        result = (
            self.pred_logits
            .cpu()
            [0, self.keep]
            .softmax(-1)
            .__gt__(.1)
            .nonzero()
            .tolist()
        )
        return result

    @cached_property
    def bboxes_scaled(self):
        # convert boxes from [0; 1] to image scales
        return (
            self.pred_boxes
            .cpu()
            [0, self.keep]
        )

    @cached_property
    def xywh(self):
        xywh = (
            self.pred_boxes
            .cpu()
            .numpy()
        )
        return xywh

    # @cached_property
    # def labels(self) -> list[str]:
    #     positive_tokens = (
    #         self.pred_logits
    #         .cpu()
    #         [0, self.keep]
    #         .softmax(-1)
    #         .__gt__(.1)
    #         .nonzero()
    #         .tolist()
    #     )
    #     predicted_spans = defaultdict(str)
    #     for tok in positive_tokens:
    #         item, pos = tok
    #         if pos < 255:
    #             span = memory_cache["tokenized"].token_to_chars(0, pos)
    #             predicted_spans[item] += " " + caption[span.start:span.end]
    #
    #     labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]
    #     return labels
    #


class MDETR(
    mdetr.MDETR
):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        return result

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        result = super().forward(samples, captions, encode_and_save, memory_cache)
        if memory_cache is not None:
            result = Result(**result)
        return result

    @classmethod
    # def from_elsa(cls):
    # mimic the params of torch.hub.load
    def from_elsa(
            cls,
            repo_or_dir: str = 'ashkamath/mdetr:main',
            model: str = 'mdetr_efficientnetB5',
            pretrained: bool = True,
            return_postprocessor: bool = True,
            **kwargs,
    ) -> Self:
        """
        # Oftentimes CoPilot suggests # this is a hack which makes me doubt my code
        # But this time, it is actually a hack!
        """
        model, postprocessor = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            pretrained=pretrained,
            return_postprocessor=return_postprocessor,
            **kwargs,
        )
        obj = object.__new__(cls)
        obj.__dict__.update(model.__dict__)
        model: Self = obj
        model = model.cuda()
        model.eval()
        return model


def build(args):
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    qa_dataset = None
    if args.do_qa:
        assert not (
                ("clevr" in args.combine_datasets or "clevr_question" in args.combine_datasets)
                and "gqa" in args.combine_datasets
        ), "training GQA and CLEVR simultaneously is not supported"
        assert (
                "clevr_question" in args.combine_datasets
                or "clevr" in args.combine_datasets
                or "gqa" in args.combine_datasets
        ), "Question answering require either gqa or clevr dataset"
        qa_dataset = "gqa" if "gqa" in args.combine_datasets else "clevr"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        qa_dataset=qa_dataset,
        split_qa_heads=args.split_qa_heads,
        predict_final=args.predict_final,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.do_qa:
        if args.split_qa_heads:
            weight_dict["loss_answer_type"] = 1 * args.qa_loss_coef
            if qa_dataset == "gqa":
                weight_dict["loss_answer_cat"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_attr"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_rel"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_obj"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_global"] = 1 * args.qa_loss_coef
            else:
                weight_dict["loss_answer_binary"] = 1
                weight_dict["loss_answer_attr"] = 1
                weight_dict["loss_answer_reg"] = 1

        else:
            weight_dict["loss_answer_total"] = 1 * args.qa_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None

    if args.do_qa:
        if qa_dataset == "gqa":
            qa_criterion = QACriterionGQA(split_qa_heads=args.split_qa_heads)
        elif qa_dataset == "clevr":
            qa_criterion = QACriterionClevr()
        else:
            assert False, f"Invalid qa dataset {qa_dataset}"
        qa_criterion.to(device)
    else:
        qa_criterion = None
    return model, criterion, contrastive_criterion, qa_criterion, weight_dict
