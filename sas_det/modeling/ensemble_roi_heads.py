# Copyright (c) NEC Laboratories America, Inc.
import os
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

from detectron2.modeling import ResNet, build_box_head, build_keypoint_head, build_mask_head, build_backbone
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY #, select_foreground_proposals
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels

from .roi_heads.clip_roi_heads import CLIPRes5ROIHeads, select_foreground_proposals
from .ensemble_fast_rcnn import EnsembleFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class EnsembleCLIPRes5ROIHeads(CLIPRes5ROIHeads):
    """
    Split-And-Fusion (SAF) head
    """

    @configurable
    def __init__(
        self,
        *,
        text_box_head: nn.Module = None,
        # use_offline_pl = False,
        **kwargs,
    ):
        """See detectron2.modeling.roi_heads.CLIPRes5ROIHeads
        """
        super().__init__(**kwargs)

        # will do box reg on this head. image head comes from attnpool in CLIP
        self.text_box_head = text_box_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        
        in_features = ret["in_features"]
        # If EnsembleCLIPRes5ROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        # build box head
        closed_head_name = cfg.MODEL.ROI_BOX_HEAD.NAME
        assert closed_head_name == "CLIP_BOX_HEAD"

        # use [ModifiedResNet.layer4, ModifiedResNet.attnpool] as box head
        backbone = build_backbone(cfg)  # CLIP backbone
        # import ipdb
        # ipdb.set_trace()
        # load weights if given
        if cfg.MODEL.WEIGHTS != "":
            ckpt_states = torch.load(cfg.MODEL.WEIGHTS)
            assert 'model' in ckpt_states.keys()
            ckpt_states = ckpt_states['model']

            # len('backbone.') == 9
            ckpt_states_backbone = {k[9:]: v for k, v in ckpt_states.items() if k.startswith('backbone')}
            backbone.load_state_dict(ckpt_states_backbone, strict=True)

        cur_res5 = copy.deepcopy(backbone.layer4)
        cur_attnpool = copy.deepcopy(backbone.attnpool)
        text_box_head = nn.Sequential(
            cur_res5,
            cur_attnpool
        )

        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 8
        box_predictor = EnsembleFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        # # get continuous novel cat ids from the external file
        # if os.path.exists(cfg.MODEL.OVD.CATEGORY_INFO):
        #     cat_info = json.load(open(cfg.MODEL.OVD.CATEGORY_INFO, "r"))
        #     base_cat_ids = cat_info["base_cat_ids"]
        # else:
        #     base_cat_ids = None

        if cfg.MODEL.MASK_ON:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        ret.update({
            'box_predictor': box_predictor,
            #
            'text_box_head': text_box_head,
            # 'base_cat_ids': base_cat_ids,
            # 'use_offline_pl': cfg.MODEL.ENSEMBLE.USE_OFFLINE_PL,
        })
        return ret

    # def prepare_running(self):
    #     '''
    #     anything that we want just once before forward, e.g. clamp temperature that may be used in multiple places
    #     '''
    
    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, bg_label=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        NOTE: Changes:
            1. background label as a parameter. GT and PLs may have different num of classes
        """
        if bg_label is None:
            bg_label = self.num_classes

        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = bg_label
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + bg_label

        # only sample fg proposals to train recognition branch (ref to subsample_labels)
        if self.only_sample_fg_proposals:
            if has_gt:
                positive = nonzero_tuple((gt_classes != -1) & (gt_classes != bg_label))[0]
                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)
                # randomly select positive and negative examples
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                sampled_idxs = positive[perm1]
            else:  # no gt, only keep 1 bg proposal to fill the slot
                sampled_idxs = torch.zeros_like(matched_idxs[0:1])
            return sampled_idxs, gt_classes[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, bg_label
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], bg_label=None,
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        NOTE: Changes:
            1. background label as a parameter. GT and PLs may have different num of classes
        """
        # import ipdb
        # ipdb.set_trace()
        if bg_label is None:
            bg_label = self.num_classes

        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, bg_label=bg_label
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == bg_label).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        #print("num_fg: {}; num_bg: {}".format(num_fg_samples, num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None, res5=None, attnpool=None, ann_type='box'):
        """
        See :meth:`ROIHeads.forward`.
            targets: include cats for PLs
            res5: 
            attnpool: 
            ann_type: must be 'box'
        """
        if not self.training:
            return self.inference(images, features, proposals, res5, attnpool)

        losses = {}
        del images

        assert ann_type == 'box', 'only support box not %s'%ann_type
        assert targets

        ## image head forward (open-branch)
        # import ipdb
        # ipdb.set_trace()
        # size of concept pool as bg_label
        proposals_img = self.label_and_sample_proposals(proposals, targets, bg_label=self.box_predictor.image_head_concepts_num)
        box_feats_img = self._shared_roi_transform(
            [features[f] for f in self.in_features], [x.proposal_boxes for x in proposals_img], res5
        )     
        # pooler_res x pooler_res -(res5)-> 7 x 7

        if attnpool:  # att pooling
            att_feats = attnpool(box_feats_img)
        else: # mean pooling
            att_feats = box_feats_img.mean(dim=[2, 3])

        # import ipdb
        # ipdb.set_trace()
        predictions_img = self.box_predictor.image_head_forward(att_feats, use_txt_pool=True)     # detection_scores, box_delta
        losses_img = self.box_predictor.image_head_losses(predictions_img, proposals_img)
        losses.update(losses_img)

        ## text head forward (close-branch)
        # remove PLs from targets. Assume cats for PLs >= self.num_classes
        targets_base = []
        for cur_tar in targets:
            cur_gt_classes = cur_tar.gt_classes
            # matCompare = cur_gt_classes[:, None] == self.base_cat_ids[None, :].to(cur_gt_classes.device)
            # base_cat_mask = matCompare.sum(dim=1).to(torch.bool)
            base_cat_mask = cur_gt_classes < self.num_classes   # num of class for Base
            targets_base.append(cur_tar[base_cat_mask])

        # import ipdb
        # ipdb.set_trace()
        # sample proposals again, self.num_classes as bg_label
        proposals_base = self.label_and_sample_proposals(proposals, targets_base, bg_label=self.num_classes)
        box_features = self.pooler([features[f] for f in self.in_features], [x.proposal_boxes for x in proposals_base]) # pooler_res x pooler_res
        
        # import ipdb
        # ipdb.set_trace()
        # use features after res5 as box_features, in line with original regionclip
        box_features = self.text_box_head[0](box_features)  # CLIP res5, do this following self._shared_roi_transform()
        features_txt_head = self.text_box_head[1](box_features) # CLIP attention pool

        predictions = self.box_predictor.text_head_forward(features_txt_head)     # detection_scores, box_delta
        losses_txt_head = self.box_predictor.text_head_losses(predictions, proposals_base)
        losses.update(losses_txt_head)

        del features
        del targets
        del targets_base

        # import ipdb
        # ipdb.set_trace()
        if self.mask_on:
            # # ignore mask w/ confidence < 1.0 (usually are PLs)
            # for pp in proposals_base:
            #     if pp.has('gt_confidence'):
            #         pp.gt_use_seg = (pp.gt_confidence >= 1.0)

            proposals, fg_selection_masks = select_foreground_proposals(
                proposals_base, self.num_classes
            )
            # Since the ROI feature transform is shared between boxes and masks,
            # we don't need to recompute features. The mask loss is only defined
            # on foreground proposals, so we need to select out the foreground
            # features.
            mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
            del box_features
            losses.update(self.mask_head(mask_features, proposals))

        return [], losses

    def forward_with_given_boxes(self, features, instances):
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
        NOTE: res5 in RegionCLIP is not used here
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            box_features = self.pooler(features, [x.pred_boxes for x in instances]) # pooler_res x pooler_res
            # use features after res5 as box_features, in line with original regionclip
            box_features = self.text_box_head[0](box_features)  # CLIP res5, do this following self._shared_roi_transform()

            return self.mask_head(box_features, instances)
        else:
            return instances
    
    def get_ovd_pseudo_labels(self, features, proposals, res5=None, attnpool=None, threshold=0.9, nms_thres=0.6):
        """
        will be used in teacher model
        """
        # import ipdb
        # ipdb.set_trace()
        with torch.no_grad():
            # # shared box features
            # # box_features = self._shared_roi_transform(
            # #     [features[f] for f in self.in_features], [x.proposal_boxes for x in proposals], res5
            # # )
            # box_features = self.pooler([features[f] for f in self.in_features], [x.proposal_boxes for x in proposals]) # pooler_res x pooler_res

            # box_feats_img = res5(box_features)  # 7 x 7
            # # image head predictions
            # if attnpool:  # att pooling
            #     att_feats = attnpool(box_feats_img)
            # else: # mean pooling
            #     att_feats = box_feats_img.mean(dim=[2, 3])
            att_feats = self.get_region_features(features, proposals, res5=res5, attnpool=attnpool)
            scores_img, proposal_deltas_img = self.box_predictor.image_head_forward(att_feats, use_txt_pool=True)     # detection_scores, proposal_deltas_img may be None

            # use scores from image head only. May use scores_txt later
            if proposal_deltas_img is None:
                # text head predictions 
                box_features = self.pooler([features[f] for f in self.in_features], [x.proposal_boxes for x in proposals]) # pooler_res x
                features_txt_head = self.text_box_head(box_features)

                scores_txt, proposal_deltas = self.box_predictor.text_head_forward(features_txt_head)     # detection_scores [N x base_num], box_delta
                predictions = (scores_img, proposal_deltas)
            else:
                predictions = (scores_img, proposal_deltas_img)

            pl_instances, _ = self.box_predictor.inference_for_pseudo_label(predictions, proposals, threshold=threshold, nms_thres=nms_thres)

            # import ipdb
            # ipdb.set_trace()
            # if mask predictions is required
            pl_instances = self.forward_with_given_boxes(features, pl_instances)
        return pl_instances

    def inference(self, images, features, proposals, res5=None, attnpool=None):
        # import ipdb
        # ipdb.set_trace()
        # shared box features
        box_features = self.pooler([features[f] for f in self.in_features], [x.proposal_boxes for x in proposals]) # pooler_res x pooler_res

        box_feats_img = res5(box_features)  # 7 x 7
        # image head predictions
        if attnpool:  # att pooling
            att_feats = attnpool(box_feats_img)
        else: # mean pooling
            att_feats = box_feats_img.mean(dim=[2, 3])
        predictions_img = self.box_predictor.image_head_forward(att_feats, use_txt_pool=False)     # detection_scores

        # text head predictions
        features_txt_head = self.text_box_head(box_features)
        predictions = self.box_predictor.text_head_forward(features_txt_head)     # detection_scores, box_delta

        pred_instances, _ = self.box_predictor.inference(predictions, predictions_img, proposals)
        pred_instances = self.forward_with_given_boxes(features, pred_instances)
        return pred_instances, {}

