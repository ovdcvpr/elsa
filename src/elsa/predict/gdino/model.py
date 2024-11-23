from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import *
from typing import *

from elsa.local import local
from open_groundingdino.tools.inference_on_a_image import *
from open_groundingdino.models.GroundingDINO import groundingdino
from open_groundingdino.models.GroundingDINO.groundingdino import *
from open_groundingdino.models.GroundingDINO.bertwarper import *


@dataclass
class Result:
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    input_ids: torch.Tensor
    offset_mapping: torch.Tensor

    def __post_init__(self):
        offset_mapping = (
            self.offset_mapping
            .squeeze()
            .cpu()
            # .detach()
            .numpy()
        )
        if len(offset_mapping.shape) == 3:
            offset_mapping = offset_mapping[0]
        self.offset_mapping = offset_mapping

    @cached_property
    def icol(self):
        offset_mapping = self.offset_mapping
        icol = np.arange(len(offset_mapping))
        loc = icol > 0
        loc &= icol < (len(offset_mapping) - 2)
        icol = icol[loc]
        return icol

    @cached_property
    def confidence(self):
        confidence = (
            self.pred_logits
            .sigmoid()
            [:, :, self.icol]
            .cpu()
            .numpy()
        )
        return confidence

    @cached_property
    def xywh(self):
        xywh = (
            self.pred_boxes
            .cpu()
            .numpy()
        )
        return xywh


class GroundingDINO(groundingdino.GroundingDINO):
    @classmethod
    def from_elsa(
            cls,
            config: str = None,
            checkpoint: str = None,
    ) -> Self:
        # groundingdino/models/GroundingDINO/backbone/swin_transformer.py:448
        message = ".*use_reentrant parameter should be passed explicitly.*"
        warnings.filterwarnings("ignore", message=message)

        # groundingdino/ models / GroundingDINO / bertwarper.py:110
        message = ".*The `device` argument is deprecated.*"
        warnings.filterwarnings('ignore', message=message)

        if config is None:
            config = local.predict.gdino.config
        if checkpoint is None:
            checkpoint = local.predict.gdino.checkpoint
        args = SLConfig.fromfile(config)
        args.device = 'cuda'
        assert 'elsa_groundingdino' in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get('elsa_groundingdino')
        model = build_func(args)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def forward(
            self,
            samples: NestedTensor,
            targets: List = None,
            **kw
    ) -> Result:
        """
        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        tokenized = self.tokenizer(
            captions,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(samples.device)

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized,
            self.specical_tokens,
            self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            # text_self_attention_masks = text_self_attention_masks[
            #                             :, : self.max_text_len, : self.max_text_len
            #                             ]
            text_self_attention_masks = text_self_attention_masks \
                [:, :, self.max_text_len, :self.max_text_len]

            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v
                for k, v in tokenized.items()
                if k not in ["attention_mask", "offset_mapping"]
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        layers = [
            embed(hs, text_dict)
            for embed, hs in zip(self.class_embed, hs)
        ]
        outputs_class = torch.stack(layers)
        out = Result(
            pred_logits=outputs_class[-1],
            pred_boxes=outputs_coord_list[-1],
            input_ids=tokenized.input_ids,
            offset_mapping=tokenized.offset_mapping,
        )

        return out


@MODULE_BUILD_FUNCS.registe_with_name(module_name='elsa_groundingdino')
def build_groundingdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model
