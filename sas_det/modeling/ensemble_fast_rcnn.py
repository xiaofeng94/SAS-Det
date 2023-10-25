# Copyright (c) NEC Laboratories America, Inc.
import os
import json
import copy
import logging
import itertools
from typing import Dict, List, Tuple, Union
import math
from scipy.optimize import linear_sum_assignment
import numpy as np

import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
# from detectron2.layers.soft_nms import batched_soft_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
# from detectron2.utils.comm import MILCrossEntropy
# from detectron2.modeling.roi_heads import FastRCNNOutputLayers
# from .roi_heads.fast_rcnn import fast_rcnn_inference, _log_classification_stats

from .roi_heads.clip_fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference, _log_classification_stats
# logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class EnsembleFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        *,
        image_head_concepts = None,
        #
        base_cat_ids = None,
        alpha = 0.2,    # fusion weight as F-VLM
        beta = 0.45,    # fusion weight as F-VLM
        use_img_head_box_reg = False,
        **kwargs,
    ):
        """See FastRCNNOutputLayers
        NOTE: when use init_concepts_path, set init_temperature to a low value 
        """
        super().__init__(**kwargs)
        assert image_head_concepts
        assert os.path.exists(image_head_concepts)
        open_txt_emb = torch.load(image_head_concepts) # num x dim
        self.register_buffer("image_head_concepts", open_txt_emb, False) # [#concepts, d]
        self.image_head_concepts_num = self.image_head_concepts.shape[0]
        
        # import ipdb
        # ipdb.set_trace()
        self.base_cat_ids = base_cat_ids    # used for inference w/ ensemble

        self.alpha = alpha
        self.beta = beta

        self.bbox_pred_img_head = copy.deepcopy(self.bbox_pred) if use_img_head_box_reg else None

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

        # get continuous novel cat ids from the external file
        test_category_info = cfg.MODEL.ENSEMBLE.TEST_CATEGORY_INFO
        assert test_category_info
        assert os.path.exists(test_category_info)
        cat_info = json.load(open(test_category_info, "r"))
        base_cat_ids = torch.tensor(cat_info["base_cat_ids"], dtype=torch.long)

        # import ipdb
        # ipdb.set_trace()
        ret.update({
            "image_head_concepts": cfg.MODEL.CLIP.CONCEPT_POOL_EMB,
            #
            "base_cat_ids": base_cat_ids,
            "alpha": cfg.MODEL.ENSEMBLE.ALPHA,
            "beta": cfg.MODEL.ENSEMBLE.BETA,
            "use_img_head_box_reg": cfg.MODEL.ENSEMBLE.USE_IMG_HEAD_BOX_REG,
        })
        return ret

    # def forward(self, x):
    #     """
    #     Args:
    #         x: per-region features of shape (N, ...) for N bounding boxes to predict.

    #     Returns:
    #         (Tensor, Tensor):
    #         First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
    #         scores for K object categories and 1 background class.

    #         Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
    #         or (N,4) for class-agnostic regression.
    #     """
    #     pass

    def text_head_forward(self, x):
        """
        Closed-branch
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # import ipdb
        # ipdb.set_trace()
        normalized_x = F.normalize(x, p=2.0, dim=1)
        # open-set inference enabled
        if not self.training and self.test_cls_score is not None: 
            cls_scores = normalized_x @ F.normalize(self.test_cls_score.weight, p=2.0, dim=1).t()
            if self.use_bias:
                cls_scores += self.test_cls_score.bias
        # training or closed-set model inference
        else: 
            cls_scores = normalized_x @ F.normalize(self.cls_score.weight, p=2.0, dim=1).t()
            if self.use_bias:
                cls_scores += self.cls_score.bias
        
        # background class (zero embeddings)
        bg_score = self.cls_bg_score(normalized_x)
        if self.use_bias:
            bg_score += self.cls_bg_score.bias

        scores = torch.cat((cls_scores, bg_score), dim=1)
        scores = scores / self.temperature
        
        # box regression
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def text_head_losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`text_head_forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # assume all from GT for text head
        cls_gt_conf = None
        # some PLs for base from Objects365 (not LVIS)
        box_gt_conf = None
        if self.box_confidence_thres > 0:
            gt_confidence = []
            for pp in proposals:
                if pp.has('gt_confidence'):
                    gt_confidence.append(pp.gt_confidence)
                else:
                    # all background proposals may not have `gt_confidence`
                    gt_confidence.append( torch.ones_like(pp.objectness_logits) )
            gt_confidence = cat(gt_confidence, dim=0)
            box_gt_conf = gt_confidence

        # import ipdb
        # ipdb.set_trace()
        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        # import ipdb
        # ipdb.set_trace()
        # loss weights
        if self.cls_loss_weight is not None and self.cls_loss_weight.device != scores.device:
            self.cls_loss_weight = self.cls_loss_weight.to(scores.device)
        if self.focal_scaled_loss is not None:
            loss_cls = self.focal_loss(scores, gt_classes, bg_label=self.num_classes, gt_confidence=cls_gt_conf, gamma=self.focal_scaled_loss)
        else:    
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean") if self.cls_loss_weight is None else \
                       cross_entropy(scores, gt_classes, reduction="mean", weight=self.cls_loss_weight)
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, bg_label=self.num_classes, gt_confidence=box_gt_conf
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def image_head_forward(self, x, use_txt_pool=False):
        """
        Open-branch.
            use_txt_pool: used for PL generation and training
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        normalized_x = F.normalize(x, p=2.0, dim=1)
        if use_txt_pool:
            cls_scores = normalized_x @ F.normalize(self.image_head_concepts, p=2.0, dim=1).t()
            # NOTE: use_bias should be False. self.image_head_concepts is a buffer w/o bias
            if self.use_bias:
                cls_scores += self.cls_score.bias
        elif not self.training and self.test_cls_score is not None: 
            cls_scores = normalized_x @ F.normalize(self.test_cls_score.weight, p=2.0, dim=1).t()
            if self.use_bias:
                cls_scores += self.test_cls_score.bias
        else:
            raise NotImplementedError

        # background class (zero embeddings)
        bg_score = self.cls_bg_score(normalized_x)
        if self.use_bias:
            bg_score += self.cls_bg_score.bias

        scores = torch.cat((cls_scores, bg_score), dim=1)
        scores = scores / self.temperature

        # box regression if any
        if self.bbox_pred_img_head:
            proposal_deltas = self.bbox_pred_img_head(x)
        else:
            proposal_deltas = None
        return scores, proposal_deltas

    def image_head_losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`image_head_forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # load gt_confidence if use PLs
        box_gt_conf = None
        cls_gt_conf = None
        # import ipdb
        # ipdb.set_trace()
        if self.box_confidence_thres > 0 or self.use_confidence_weight:
            gt_confidence = []
            for pp in proposals:
                if pp.has('gt_confidence'):
                    gt_confidence.append(pp.gt_confidence)
                else:
                    # all background proposals may not have `gt_confidence`
                    gt_confidence.append( torch.ones_like(pp.objectness_logits) )
            gt_confidence = cat(gt_confidence, dim=0)

            box_gt_conf = gt_confidence
            
            if self.use_confidence_weight:
                # set confidence of PLs to 0.2 (same weight as background, 0.2 for COCO, 0.8 for LVIS)
                cls_gt_conf = gt_confidence.clone()
                cls_gt_conf[cls_gt_conf < 1.0] = self.cls_loss_weight[-1].item() 

        # import ipdb
        # ipdb.set_trace()
        # loss weights
        # if self.cls_loss_weight is not None and self.cls_loss_weight.device != scores.device:
        #     self.cls_loss_weight = self.cls_loss_weight.to(scores.device)
        cls_loss_weight = None     # self.cls_loss_weight is setup for training on base. disable it for now
        if self.focal_scaled_loss is not None:
            loss_cls = self.focal_loss(scores, gt_classes, bg_label=self.image_head_concepts_num, gt_confidence=cls_gt_conf, gamma=self.focal_scaled_loss)
        else:    
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean") if cls_loss_weight is None else \
                       cross_entropy(scores, gt_classes, reduction="mean", weight=cls_loss_weight)
        losses = {
            "loss_cls_img_head": loss_cls,
        }

        if proposal_deltas is not None:
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat(
                    [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                    dim=0,
                )
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes, bg_label=self.image_head_concepts_num, gt_confidence=box_gt_conf)
            losses.update({"loss_box_reg_img_head": loss_box_reg})

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def focal_loss(self, inputs, targets, bg_label=None, gt_confidence=None, gamma=0.5, reduction="mean"):
        """Inspired by RetinaNet implementation
                targets: (box_num, )
                gt_confidence: (box_num, ), loss confidence for targets
            NOTE: Changes: 1. new param bg_label
        """
        if bg_label is None:
            bg_label = self.num_classes
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient
        
        # focal scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        loss = ce_loss * ((1 - p_t) ** gamma)

        # import ipdb
        # ipdb.set_trace()
        # bg loss weight / confidence
        loss_weight = torch.ones(loss.size(0)).to(p.device)

        # confidence scores
        if  gt_confidence is not None:
            loss_weight.copy_(gt_confidence)

        # bg loss weight
        if self.cls_loss_weight is not None:
            bg_mask = (targets == bg_label)
            loss_weight[bg_mask] = self.cls_loss_weight[-1].item()
            # loss_weight = self.cls_loss_weight[targets] # set weight for each predictions based on the category

        loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, bg_label=None, gt_confidence=None):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        NOTE: Changes: 1. new param bg_label
        """
        if bg_label is None:
            bg_label = self.num_classes

        box_dim = proposal_boxes.shape[1]  # 4 or 5

        # import ipdb
        # ipdb.set_trace()
        if self.box_confidence_thres > 0 and gt_confidence is not None:
            # when using pseudo labels, only computed on high confidence boxes
            temp_mask = (gt_classes >= 0) & (gt_classes < bg_label) & (gt_confidence >= self.box_confidence_thres)
            fg_inds = nonzero_tuple(temp_mask)[0]
        else:
            # Regression loss is only computed for foreground proposals (those matched to a GT)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_label))[0]

        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, bg_label, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        if self.box_confidence_thres > 0 and gt_confidence is not None:
            # divide the num of GT + high confidence PLs
            cur_divisor = (gt_confidence >= self.box_confidence_thres).sum()
            return loss_box_reg / max(cur_divisor, 1.0)  # return 0 if empty
        else:
            return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(
        self, 
        predictions: Tuple[torch.Tensor, torch.Tensor],     # text head outputs, (scores, proposal_deltas)
        predictions_img: Tuple[torch.Tensor, torch.Tensor],    # image head outputs, (scores, proposal_deltas)
        proposals: List[Instances], 
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        # scores from text head
        boxes = self.predict_boxes(predictions, proposals)
        scores_txt = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # scores from image head
        # boxes_img = self.predict_boxes(predictions_img, proposals)  # not used for now
        scores_img = self.predict_probs(predictions_img, proposals)

        # import ipdb
        # ipdb.set_trace()
        total_num_classes = scores_img[0].shape[-1] # base + novel + bg
        base_index = torch.bincount(self.base_cat_ids)
        base_index = F.pad(base_index, (0, total_num_classes - len(base_index))).bool().to(scores_img[0].device) # treat bg as novel cls
        
        # ensemble, //TODO: handle bg w/ a different weight
        scores = []
        for cur_scores_txt, cur_scores_img in zip(scores_txt, scores_img):
            cur_scores = torch.where(base_index[None, :], cur_scores_txt**(1 - self.alpha) * cur_scores_img**self.alpha, cur_scores_txt**(1 - self.beta) * cur_scores_img**self.beta)
            scores.append(cur_scores)

        # optional: multiply class scores with RPN scores 
        scores_bf_multiply = scores  # as a backup for visualization purpose
        if self.multiply_rpn_score and not self.training:
            # will be used on LVIS
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]
        
        # import ipdb
        # ipdb.set_trace()
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.soft_nms_enabled,
            self.soft_nms_method,
            self.soft_nms_sigma,
            self.soft_nms_prune,
            self.test_topk_per_image,
            scores_bf_multiply = scores_bf_multiply,
            vis = True if self.vis else False,
        )