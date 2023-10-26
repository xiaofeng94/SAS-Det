# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) NEC Laboratories America, Inc.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import itertools
import copy
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.utils import comm
# from detectron2.utils.comm import gather_tensors

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
# from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, ROIHeads

from .clip_fast_rcnn import FastRCNNOutputLayers

## used for mask_head, will ignore pls with masks
def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    # import ipdb
    # ipdb.set_trace()
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    with_gt_use_seg = proposals[0].has("gt_use_seg") # for PLs
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        if with_gt_use_seg:
            # for PLs
            gt_use_seg = proposals_per_image.gt_use_seg
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (gt_use_seg > 0.)
        else:
            # default for detectron2
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class CLIPRes5ROIHeads(ROIHeads):
    """
    Created for CLIP ResNet. This head uses the last resnet layer from backbone.
    Extended from Res5ROIHeads in roi_heads.py
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: None,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        #
        weak_loss_type='contrastive',
        weak_loss_weight=-1.0,
        num_regions_per_img=32,
        box_select_thres=0.97,
        concept_thres=0.1, 
        pseudo_loss_weight=-1.0,
        pseudo_use_distill=False,
        pseudo_use_contrastive=False,
        #
        # dataset_bs=None,
        text_emb_dim=1024,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        # if isinstance(res5, (list, tuple)):
        #     res5 = nn.Sequential(*res5)
        self.res5 = res5  #  None, this head uses the res5 from backbone
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

        # by zsy
        self.weak_loss_type = weak_loss_type
        self.weak_loss_weight = weak_loss_weight
        self.num_regions_per_img = num_regions_per_img
        self.box_select_thres = box_select_thres

        self.concept_thres = concept_thres
        self.pseudo_loss_weight = pseudo_loss_weight
        self.pseudo_use_distill = pseudo_use_distill
        self.pseudo_use_contrastive = pseudo_use_contrastive

        self.logger = logging.getLogger("detectron2.trainer")

        # # for ita loss
        # self.dataset_bs = dataset_bs
        # # self.det_cap_data_ratio = None if dataset_bs is None else int(dataset_bs[1]/dataset_bs[0])
        self.text_emb_dim = text_emb_dim

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        # if not inspect.ismethod(cls._build_res5_block):
        #     logger.warning(
        #         "The behavior of _build_res5_block may change. "
        #         "Please do not depend on private methods."
        #     )
        #     cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = None, cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 8 # cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        ret.update({
            'weak_loss_type': cfg.MODEL.WEAK_LOSS.WEAK_LOSS_TYPE,
            'weak_loss_weight': cfg.MODEL.WEAK_LOSS.WEAK_LOSS_WEIGHT,
            "num_regions_per_img": cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS,
            "box_select_thres": cfg.MODEL.WEAK_LOSS.BOX_SELECT_THRES,
            # regionclip PLs
            "concept_thres": cfg.MODEL.CLIP.CONCEPT_THRES,
            "pseudo_loss_weight": cfg.MODEL.WEAK_LOSS.PSEUDO_LOSS_WEIGHT,
            "pseudo_use_distill": cfg.MODEL.WEAK_LOSS.PSEUDO_USE_DISTILL,
            "pseudo_use_contrastive": cfg.MODEL.WEAK_LOSS.PSEUDO_USE_CONTRASTIVE,
            #
            # "dataset_bs": cfg.DATALOADER.DATASET_BS,
            "text_emb_dim": cfg.MODEL.CLIP.TEXT_EMB_DIM, 
        })
        return ret

    def _shared_roi_transform(self, features, boxes, backbone_res5):
        x = self.pooler(features, boxes)
        return backbone_res5(x)

    def _get_top_proposals(self, proposals, topk=32, thres=0.9):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        # topk
        if topk > 0:
            proposals = [p[:topk] for p in proposals]
        # thresholding
        if thres > 0:
            proposals_thres = []
            for pps in proposals:
                # remove small boxes
                pps_no_small = pps[pps.proposal_boxes.area() > 81]
                pps_thres = pps_no_small[pps_no_small.objectness_logits.sigmoid() > thres]
                if len(pps_thres) > 10:
                    proposals_thres.append(pps_thres)
                else:
                    proposals_thres.append(pps[:10])
            proposals = proposals_thres
        # no gradient for box
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
        return proposals

    def prepare_running(self):
        '''
        anything that we want just once before forward, e.g. clamp temperature that may be used in multiple places
        '''
        pass

    def forward(self, images, features, proposals, targets=None, res5=None, attnpool=None, ann_type='box', ema_inputs=[None, None]):
        """
        See :meth:`ROIHeads.forward`.
            res5: 
            attnpool: 
            ann_type: 
            features_m: features from ema teacher model. Used when ema_contrastive and ema_sinkhorn
            ema_inputs: teacher_backbone & teacher_roi_heads
        """
        if ann_type == 'feature_only':
            return self.get_region_features(features, proposals, res5=res5, attnpool=attnpool)

        # import ipdb
        # ipdb.set_trace()
        if self.training:
            if ann_type in ['box']:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)

                del images
                # del targets
            elif ann_type in ['caption']:
                 # select box for weak supervision
                proposals = self._get_top_proposals(proposals, topk=self.num_regions_per_img, thres=self.box_select_thres)
            else:
                raise NotImplementedError
        
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, res5
        )
        if attnpool:  # att pooling
            att_feats = attnpool(box_features)
            # predictions = self.box_predictor(att_feats)
        else: # mean pooling
            att_feats = box_features.mean(dim=[2, 3])
            # predictions = self.box_predictor(att_feats)

        if self.training:
            # gather all targets for some weak loss. Every GPU should run this to avid block
            # self.logger.info(f"[{comm.get_rank()}] BEFORE all_gather for weak_align")
            if self.weak_loss_type in ['weak_align']:
                all_att_feats = [att_feats]
                all_proposals = [proposals]
                all_targets = [targets]
            
            del features
            losses = {}

            # weak supervision for region-concept level matching
            if ann_type == 'caption':
                # weak loss
                if self.weak_loss_weight > 0:
                    if self.weak_loss_type == 'contrastive':
                        weak_loss = self.box_predictor.align_contrastive_loss(att_feats, proposals, targets, normalize=True)
                    elif self.weak_loss_type == 'sinkhorn':
                        weak_loss = self.box_predictor.align_sinkhorn_loss(att_feats, proposals, targets, normalize=True)
                    elif self.weak_loss_type in ['ema_contrastive', 'ema_sinkhorn']:
                        teacher_backbone = ema_inputs[0]
                        teacher_roi_heads = ema_inputs[1]
                        # extract visual features from teacher model
                        with torch.no_grad():
                            features_m = teacher_backbone(images.tensor)
                            region_feats_m = teacher_roi_heads.get_region_features(features_m, proposals, res5=teacher_backbone.layer4, attnpool=teacher_backbone.attnpool if attnpool else None)

                        weak_loss = self.box_predictor.ema_align_loss(att_feats, region_feats_m, proposals, targets, normalize=True, align_type=self.weak_loss_type)
                    elif self.weak_loss_type in ['weak_align']:
                        weak_loss = self.box_predictor.weak_alignment_loss(all_att_feats, all_proposals, all_targets, normalize=True)
                    else:
                        raise NotImplementedError
                    losses['loss_weak'] = weak_loss * self.weak_loss_weight

                # pseudo label loss for pretraining, used for regionclip pretraining
                if self.pseudo_loss_weight > 0:
                    teacher_backbone = ema_inputs[0]
                    teacher_roi_heads = ema_inputs[1]

                    psuedo_concept_labels = self.get_psuedo_concept_labels(teacher_backbone, teacher_roi_heads, images, proposals)
                    loss_region_distill, loss_concept_contrastive = self.box_predictor.region_concept_pseudo_loss(att_feats, psuedo_concept_labels, use_distill=self.pseudo_use_distill, use_contrastive=self.pseudo_use_contrastive)

                    if self.pseudo_use_distill:
                        losses['loss_region_distill'] = loss_region_distill * self.pseudo_loss_weight
                    if self.pseudo_use_contrastive:
                        losses['loss_concept_contrastive'] = loss_concept_contrastive * self.pseudo_loss_weight

                # fake box loss
                losses['loss_cls'] = att_feats.new_zeros([])
                losses['loss_box_reg'] = att_feats.new_zeros([])

            elif ann_type == "box":
                predictions = self.box_predictor(att_feats)     # detection_scores, box_delta
                box_losses = self.box_predictor.losses(predictions, proposals)
                losses.update(box_losses)

                # fake caption loss
                if self.weak_loss_weight > 0:
                    losses['loss_weak'] = att_feats.new_zeros([])
                if self.pseudo_loss_weight > 0:
                    if self.pseudo_use_distill:
                        losses['loss_region_distill'] = att_feats.new_zeros([])
                    if self.pseudo_use_contrastive:
                        losses['loss_concept_contrastive'] = att_feats.new_zeros([])
            else:
                raise NotImplementedError

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            predictions = self.box_predictor(att_feats)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, res5)
            return pred_instances, {}

    def get_psuedo_concept_labels(self, teacher_backbone, teacher_roi_heads, images, proposals, gt_instances=None):
        """ Input images and region proposals, return matching results from teacher model
        """
        # import ipdb
        # ipdb.set_trace()
        open_txt_emb = teacher_roi_heads.box_predictor.open_txt_emb  # with all-0 background embedding
        s_temp = teacher_roi_heads.box_predictor.matching_temp  # 0.01
        concept_thres = self.concept_thres

        with torch.no_grad():
            # extract visual features from teacher model
            features = teacher_backbone(images.tensor)
            # teacher_region_feats = teacher_roi_heads(images, features, proposals, gt_instances, res5=teacher_backbone.layer4, attnpool=teacher_backbone.attnpool, ann_type='feature_only')
            teacher_region_feats = teacher_roi_heads.get_region_features(features, proposals, res5=teacher_backbone.layer4, attnpool=teacher_backbone.attnpool)

            # match teacher visual features with teacher concept embs to create pseudo labels
            teacher_region_feats = F.normalize(teacher_region_feats, p=2, dim=-1)
            teacher_concept_emb = F.normalize(open_txt_emb, p=2, dim=-1)

            concept_scores = teacher_region_feats @ teacher_concept_emb.t()  # [#regions, #concepts]
            concept_scores = F.softmax(concept_scores / s_temp, dim=1)

            max_scores, max_inds = torch.max(concept_scores[:,:-1], dim=1)  # remove background when get pseudo labels
            keep_regions = max_scores > concept_thres  # only keep the regions that have high matching score with a concept
            if keep_regions.nonzero().size(0) == 0: # if all regions can't match to any concept
                print("clip_roi_heads: all regions can't match to any concept!")
                keep_regions = max_scores > 0.0 
            target_inds = max_inds[keep_regions]
            target_embs = open_txt_emb[target_inds] # the target embedding of student model
            label_mtx = (target_inds.view(-1, 1) == target_inds.view(1, -1)).type_as(teacher_region_feats)
            concept_scores = concept_scores[keep_regions]
            
        return concept_scores, target_inds, keep_regions, target_embs, label_mtx

    def get_ovd_pseudo_labels(self, features, proposals, res5=None, attnpool=None, threshold=0.9, nms_thres=0.6):
        # import ipdb
        # ipdb.set_trace()
        with torch.no_grad():
            region_feats = self.get_region_features(features, proposals, res5=res5, attnpool=attnpool)
            predictions = self.box_predictor(region_feats)
            pl_instances, _ = self.box_predictor.inference_for_pseudo_label(predictions, proposals, threshold=threshold, nms_thres=nms_thres)
            # if mask predictions is required
            pl_instances = self.forward_with_given_boxes(features, pl_instances, res5)
        return pl_instances

    @torch.no_grad()
    def inference_ensemble(self, images, features, proposals, res5=None, attnpool=None, ema_inputs=[None, None]):
        """
        Args:
            ema_inputs: teacher_backbone, teacher_roi_heads
        """
        region_feats= self.get_region_features(features, proposals, res5=res5, attnpool=attnpool)
        predictions = self.box_predictor(region_feats)

        # import ipdb
        # ipdb.set_trace()
        # extract visual features from teacher model
        teacher_backbone = ema_inputs[0]
        teacher_roi_heads = ema_inputs[1]

        features_m = teacher_backbone(images.tensor)
        region_feats_m = teacher_roi_heads.get_region_features(features_m, proposals, res5=teacher_backbone.layer4, attnpool=teacher_backbone.attnpool if attnpool else None)
        predictions_m = teacher_roi_heads.box_predictor(region_feats_m)

        pred_instances,_ = self.box_predictor.inference_ensemble(predictions, predictions_m, proposals)
        pred_instances = self.forward_with_given_boxes(features, pred_instances, res5)
        return pred_instances

    def forward_with_given_boxes(self, features, instances, res5=None):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances], res5)
            return self.mask_head(x, instances)
        else:
            return instances
    
    def get_region_features(self, features, proposals, res5=None, attnpool=None):
        """
        See :meth:`ROIHeads.forward`.
        NOTE: will be used to get region features for ema teacher model
        """
        proposal_boxes = [x.proposal_boxes for x in proposals] # object proposals
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes, res5
        )
        # import ipdb
        # ipdb.set_trace()
        if attnpool:  # att pooling
            att_feats = attnpool(box_features)  # box_num_batch x C x 7 x 7 --> box_num_batch x C 
            region_feats = att_feats
        else: # mean pooling
            region_feats = box_features.mean(dim=[2, 3])

        return region_feats
