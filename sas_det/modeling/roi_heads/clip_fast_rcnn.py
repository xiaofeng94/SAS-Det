# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) NEC Laboratories America, Inc.
import os
import json
import logging
import itertools
from typing import Dict, List, Tuple, Union
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
from detectron2.utils import comm

from .soft_nms import batched_soft_nms
# from vldet.modeling import SinkhornDistance

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

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

class MILCrossEntropy(nn.Module):
    """
    Multi-instance learning loss
    """
    def __init__(self):
        super(MILCrossEntropy, self).__init__()

    def forward(self, x, target, dim=-1, weights=None, avg_positives=False):
        # for numerical stability
        logits_max, _ = torch.max(x, dim=1, keepdim=True)
        logits = x - logits_max.detach()
        exp_logits = torch.exp(logits)

        # get non-zero entries off-diagonal
        # identity = torch.eye(target.shape[0]).type_as(target)
        # laplacian = 1 - (target - identity)
        probs = exp_logits / (exp_logits).sum(dim=dim, keepdim=True)
        if avg_positives:  # average the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6))
        else:  # sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim))
        if weights is not None:
            return (loss * weights).mean()
        return loss.mean()

# copy from detrex
def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        soft_nms_enabled (bool): Indicate to use soft non-maximum suppression.
        soft_nms_method: (str): One of ['gaussian', 'linear', 'hard']
        soft_nms_sigma: (float): Sigma for gaussian soft nms. Value in (0, inf)
        soft_nms_prune: (float): Threshold for pruning during soft nms. Value in [0, 1]
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, 
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune, topk_per_image, s_bf_per_img, vis
        )
        for scores_per_image, boxes_per_image, image_shape, s_bf_per_img in zip(scores, boxes, image_shapes, scores_bf_multiply)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)
        #print("cls_accuracy {:.2f}; fg_cls_accuracy {:.2f}; false_negative {:.2f}".format(num_accurate / num_instances, fg_num_accurate / num_fg, num_false_negative / num_fg))


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        scores_bf_multiply = scores_bf_multiply[valid_mask]

    scores = scores[:, :-1]
    scores_bf_multiply = scores_bf_multiply[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # import ipdb
    # ipdb.set_trace()
    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    scores_bf_multiply = scores_bf_multiply[filter_mask]

    # 2. Apply NMS for each class independently.
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
        )
        scores[keep] = soft_nms_scores   
        # scores_bf_multiply? (TBD)
        scores_bf_multiply = scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    scores_bf_multiply = scores_bf_multiply[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    if vis: # visualization: convert to the original scores before multiplying RPN scores
        result.scores = scores_bf_multiply         
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def softmax_cross_entropy_loss(self):
        """
        Deprecated
        """
        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Deprecated
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        soft_nms_enabled=False,
        soft_nms_method="gaussian",
        soft_nms_sigma=0.5,
        soft_nms_prune=0.001,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        clip_cls_emb: tuple = (False, None),
        no_box_delta: bool = False,
        bg_cls_loss_weight: None,
        multiply_rpn_score: tuple = (False, False),
        openset_test: None,
        #
        concept_pool_emb = None,
        matching_temp = 0.01,
        neg_concept_num = 10, 
        rpn_fusion_method = "none",
        box_confidence_thres = -1.0,
        use_confidence_weight = False,
        #
        ensemble_alpha = 0.5,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_prune = soft_nms_prune
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        # RegionCLIP
        self.num_classes = num_classes
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
                    
        self.use_clip_cls_emb = clip_cls_emb[0]
        if self.use_clip_cls_emb: # use CLIP text embeddings as classifier's weights
            # input_size = clip_cls_emb[3] if clip_cls_emb[2] in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads', 'PretrainRes5ROIHeads'] else input_size
            input_size = clip_cls_emb[3]
            text_emb_require_grad = False
            self.use_bias = False
            self.temperature = openset_test[2] # 0.01 is default for CLIP

            # import ipdb
            # ipdb.set_trace()
            # class embedding
            self.cls_score = nn.Linear(input_size, num_classes, bias=self.use_bias)  
            with torch.no_grad():
                if clip_cls_emb[1] is not None: # it could be None during region feature extraction
                    pre_computed_w = torch.load(clip_cls_emb[1])  # [num_classes, 1024] for RN50
                    self.cls_score.weight.copy_(pre_computed_w)
                self.cls_score.weight.requires_grad = text_emb_require_grad # freeze embeddings
                if self.use_bias:
                    nn.init.constant_(self.cls_score.bias, 0)
            
            # background embedding
            self.cls_bg_score = nn.Linear(input_size, 1, bias=self.use_bias)  
            with torch.no_grad():
                nn.init.constant_(self.cls_bg_score.weight, 0)  # zero embeddings
                self.cls_bg_score.weight.requires_grad = text_emb_require_grad
                if self.use_bias:
                    nn.init.constant_(self.cls_bg_score.bias, 0)

            # class embedding during test 
            self.test_cls_score = None
            if openset_test[1] is not None:  # openset test enabled
                pre_computed_w = torch.load(openset_test[1])  # [#openset_test_num_cls, 1024] for RN50
                self.openset_test_num_cls = pre_computed_w.size(0)
                self.test_cls_score = nn.Linear(input_size, self.openset_test_num_cls, bias=self.use_bias)  
                self.test_cls_score.weight.requires_grad = False # freeze embeddings
                with torch.no_grad():
                    self.test_cls_score.weight.copy_(pre_computed_w)
                    if self.use_bias:
                        nn.init.constant_(self.test_cls_score.bias, 0)    
        else: # regular classification layer  
            self.cls_score = nn.Linear(input_size, num_classes + 1) # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
 
        # box regression layer
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        # training options
        self.cls_loss_weight = None
        if bg_cls_loss_weight is not None:  # loss weigh for bg class
            self.cls_loss_weight = torch.ones(num_classes + 1)
            self.cls_loss_weight[-1] = bg_cls_loss_weight
        self.focal_scaled_loss = openset_test[3]  # focal scaling
        # inference options
        self.no_box_delta = no_box_delta  # box delta after regression
        self.multiply_rpn_score = multiply_rpn_score[0]
        self.vis = multiply_rpn_score[1] # if enabled, visualize scores before multiplying RPN scores
        
        if self.no_box_delta:
            for p in self.bbox_pred.parameters(): 
                p.requires_grad = False

        # build concept_pool_emb
        if concept_pool_emb:
            open_txt_emb = torch.load(concept_pool_emb) # num x dim
            open_txt_emb = torch.cat(
                [open_txt_emb, open_txt_emb.new_zeros((1, open_txt_emb.shape[1]))], dim=0
            ) # (num + 1) x dim, for background embedding
            self.register_buffer("open_txt_emb", open_txt_emb, False) # [#concepts, d]
        self.matching_temp = matching_temp
        self.neg_concept_num = neg_concept_num
        self.ignore_zero_region = False
        
        # self.sinkhorn = SinkhornDistance(eps = 1e-3, max_iter=100)
        self.rpn_fusion_method = rpn_fusion_method
        self.box_confidence_thres = box_confidence_thres
        self.use_confidence_weight = use_confidence_weight

        self.ensemble_alpha = ensemble_alpha

    @classmethod
    def from_config(cls, cfg, input_shape):
        # if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
        #     assert cfg.MODEL.CLIP.NO_BOX_DELTA is False

        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "soft_nms_enabled"      : cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            "soft_nms_method"       : cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD,
            "soft_nms_sigma"        : cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA,
            "soft_nms_prune"        : cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # RegionCLIP
            "clip_cls_emb"          : (cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER, cfg.MODEL.CLIP.TEXT_EMB_PATH, cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.CLIP.TEXT_EMB_DIM),
            "no_box_delta"          : cfg.MODEL.CLIP.NO_BOX_DELTA or cfg.MODEL.CLIP.CROP_REGION_TYPE == 'GT',
            "bg_cls_loss_weight"    : cfg.MODEL.CLIP.BG_CLS_LOSS_WEIGHT,
            "multiply_rpn_score"    : (cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE, cfg.MODEL.CLIP.VIS),
            "openset_test"          : (cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES, cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH, \
                                       cfg.MODEL.CLIP.CLSS_TEMP, cfg.MODEL.CLIP.FOCAL_SCALED_LOSS),
            # by zsy
            "concept_pool_emb"      : cfg.MODEL.CLIP.CONCEPT_POOL_EMB,
            "matching_temp"         : cfg.MODEL.CLIP.CLSS_TEMP,
            "neg_concept_num"       : cfg.MODEL.WEAK_LOSS.NEG_CONCEPT_NUM,
            "rpn_fusion_method"     : cfg.MODEL.OVD.RPN_FUSION_METHOD,
            "box_confidence_thres"  : cfg.MODEL.OVD.BOX_CONFIDENCE_THRES,   # confidence socres for box regression
            "use_confidence_weight" : cfg.MODEL.OVD.USE_CONFIDENCE_WEIGHT, 
            "ensemble_alpha"        : cfg.MODEL.OVD.ENSEMBLE_ALPHA,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # import ipdb
        # ipdb.set_trace()
        # use clip text embeddings as classifier's weights
        if self.use_clip_cls_emb: 
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
        # regular classifier
        else:  
            scores = self.cls_score(x)
        
        # box regression
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
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
            # COCO goes here
            loss_cls = self.focal_loss(scores, gt_classes, gt_confidence=cls_gt_conf, gamma=self.focal_scaled_loss)
        else:
            # LVIS goes here. //TODO: try federated loss
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean") if self.cls_loss_weight is None else \
                       cross_entropy(scores, gt_classes, reduction="mean", weight=self.cls_loss_weight)
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, box_gt_conf
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def focal_loss(self, inputs, targets, gt_confidence=None, gamma=0.5, reduction="mean"):
        """Inspired by RetinaNet implementation
                targets: (box_num, )
                gt_confidence: (box_num, ), loss confidence for targets
        """
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
            bg_mask = (targets == self.num_classes)
            loss_weight[bg_mask] = self.cls_loss_weight[-1].item()
            # loss_weight = self.cls_loss_weight[targets] # set weight for each predictions based on the category

        loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, gt_confidence=None):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5

        # import ipdb
        # ipdb.set_trace()
        if self.box_confidence_thres > 0 and gt_confidence is not None:
            # //TODO: when using pseudo labels, only computed on high confidence boxes
            temp_mask = (gt_classes >= 0) & (gt_classes < self.num_classes) & (gt_confidence >= self.box_confidence_thres)
            fg_inds = nonzero_tuple(temp_mask)[0]
        else:
            # Regression loss is only computed for foreground proposals (those matched to a GT)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]

        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
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

    # def _get_negative_concepts(self, cur_iidx, targets, all_cats_num):
    #     neg_concept_list=[]
    #     for other_concepts in targets[:cur_iidx]+targets[cur_iidx+1:]:
    #         neg_concept_list = neg_concept_list + [other_concept_id for other_concept_id in other_concepts._pos_category_ids \
    #             if other_concept_id not in targets[cur_iidx]._pos_category_ids]
    #     #//TODO: make neg_concept_list to a fixed size by padding
    #     if len(neg_concept_list) < self.neg_concept_num:
    #         # random sample some negatives
    #         rand_choice_num = self.neg_concept_num - len(neg_concept_list) + len(targets[cur_iidx]._pos_category_ids)
    #         random_neg_list = np.random.choice(all_cats_num, rand_choice_num, replace=False).tolist()
    #         for rand_neg_id in random_neg_list:
    #             if rand_neg_id not in targets[cur_iidx]._pos_category_ids:
    #                 neg_concept_list.append(rand_neg_id)
    #     return neg_concept_list

    # def align_contrastive_loss(self, proj_region, proposals, targets, normalize=True):
    #     """
    #     Inputs:
    #         proj_region: box_num x dim, no global box region
    #         open_txt_emb: open_set_num x dim
    #         temperature: 0.01 by default as CLIP
    #     """
    #     # normalize
    #     zs_weight = self.open_txt_emb    # from cfg.MODEL.CLIP.CONCEPT_POOL_EMB
    #     temperature = self.matching_temp
    #     if normalize:
    #         proj_region = F.normalize(proj_region, p=2.0, dim=1)     #  box_num x dim
    #         zs_weight = F.normalize(zs_weight, p=2.0, dim=1).permute(1,0)     # dim x open_set_num

    #     batch_size = len(proposals)
    #     num_inst_per_image = [len(p) for p in proposals]
    #     proj_region = proj_region.split(num_inst_per_image, dim=0)     
    #     loss = proj_region[0].new_zeros([1])[0] #(proj_region[0][:,0]).repeat((zs_weight.size()[1]-1,1))

    #     # import ipdb
    #     # ipdb.set_trace()
    #     for ii, (proj_r, target) in enumerate(zip(proj_region, targets)):
    #         if self.ignore_zero_region and proj_r.shape[0] <= 0:
    #             continue    # some image from cc3m may be broken and no proposals (excluding the image box) for that

    #         proj_r = proj_r.permute(1,0)     #  dim x box_num
    #         target_ids = target._pos_category_ids 

    #         with torch.no_grad():
    #             similarity_for_img = torch.mm(zs_weight.permute(1,0), proj_r) / temperature   # torch.Size([4765, 32]), open_set_num x num_box
    #             ss = similarity_for_img[:-1,:]      # remove the last one (for background)
    #             norm_similarity_for_img = ss-ss.min()
    #             ot_similarity = torch.zeros(int(len(target_ids)), proj_r.size()[1]).to(zs_weight.device)
    #             for i, ind in enumerate(target_ids):
    #                 ot_similarity[i] = norm_similarity_for_img[ind]     # get all box scores for each target id, no background id
    #             distance_for_ot = -ot_similarity    # target_id_num x num_box
    #             x_s, y_s = linear_sum_assignment(distance_for_ot.detach().cpu())    # y_s: box id
    #             x_s = [target_ids[i] for i in x_s]    # continugious id --> cat_id

    #         # neg_concept_list=[]
    #         # for other_concepts in targets[:ii]+targets[ii+1:]:
    #         #     neg_concept_list = neg_concept_list + [other_concept_id for other_concept_id in other_concepts._pos_category_ids \
    #         #         if other_concept_id not in targets[ii]._pos_category_ids]
    #         # #//TODO: make neg_concept_list to a fixed size by padding
    #         # if len(neg_concept_list) < 10:
    #         #     # random sample some negatives
    #         #     random_neg_list = np.random.choice(zs_weight.shape[1], 10+len(targets[ii]._pos_category_ids), replace=False).tolist()
    #         #     for rand_neg_id in random_neg_list:
    #         #         if rand_neg_id not in targets[ii]._pos_category_ids:
    #         #             neg_concept_list.append(rand_neg_id)
    #         neg_concept_list = self._get_negative_concepts(ii, targets, zs_weight.shape[1])
    #         neg_txt_emb = zs_weight[:, neg_concept_list]

    #         # import ipdb
    #         # ipdb.set_trace()
    #         for xx_s, yy_s in zip(x_s, y_s):
    #             match_num = 0
    #             # proposal_feature = torch.zeros_like(proj_r[:,0])
    #             # word_features = torch.zeros_like(zs_weight)
    #             word_features = zs_weight.new_zeros(zs_weight.shape[0], 1+len(neg_concept_list))
    #             proposal_feature = proj_r[:, yy_s]
    #             word_features[:, match_num] = zs_weight[:, xx_s]
    #             match_num = match_num + 1
    #             word_features[:, match_num:len(neg_concept_list)+match_num] = neg_txt_emb
    #             match_num = match_num + len(neg_concept_list)
    #             optimize_matrix = word_features[:,:match_num].permute(1,0) @ proposal_feature / temperature     # (match_num,)
    #             target_matrix = torch.zeros_like(optimize_matrix).to(zs_weight.device)
    #             target_matrix[0] = 1

    #             # import ipdb
    #             # ipdb.set_trace()
    #             # loss += torch.sum(F.binary_cross_entropy_with_logits(optimize_matrix, target_matrix, reduction='none'))
    #             loss += F.cross_entropy(optimize_matrix, target_matrix, reduction='sum')

    #     return loss / batch_size

    # def align_sinkhorn_loss(self, proj_region, proposals, targets, normalize=True):
    #     """
    #     Inputs:
    #         proj_region: box_num x dim, no global box region
    #         openset_txt_em: open_set_num x dim
    #         temperature: 0.01 by default as CLIP
    #         //TODO: check implementation
    #     """
    #     # normalize
    #     zs_weight = self.open_txt_emb    # from cfg.MODEL.CLIP.CONCEPT_POOL_EMB
    #     temperature = self.matching_temp
    #     if normalize:
    #         proj_region = F.normalize(proj_region, p=2.0, dim=1)     #  box_num x dim
    #         zs_weight = F.normalize(zs_weight, p=2.0, dim=1).permute(1,0)     # dim x open_set_num

    #     batch_size = len(proposals)
    #     num_inst_per_image = [len(p) for p in proposals]
    #     proj_region = proj_region.split(num_inst_per_image, dim=0)     
    #     loss = proj_region[0].new_zeros([1])[0]     # (proj_region[0][:,0]).repeat((zs_weight.size()[1]-1,1))

    #     # import ipdb
    #     # ipdb.set_trace()
    #     for ii, (proj_r, target) in enumerate(zip(proj_region, targets)):
    #         if self.ignore_zero_region and proj_r.shape[0] <= 0:
    #             continue   # some image from cc3m may be broken and no proposals (excluding the image box) for that

    #         proj_r = proj_r.permute(1,0)     # dim x box_num
    #         cur_num_regions = proj_r.shape[1]
    #         target_ids = target._pos_category_ids 

    #         with torch.no_grad():
    #             similarity_for_img = torch.mm(zs_weight.permute(1,0), proj_r) / temperature   # [4765 x box_unm (32)]
    #             ot_similarity = torch.zeros(int(len(target_ids))+1, proj_r.size()[1]).to(zs_weight.device)
    #             for i, ind in enumerate(target_ids):
    #                 ot_similarity[i] = similarity_for_img[ind]  # fetch scores for each target_id
    #             ot_similarity[-1]  = similarity_for_img[-1]     # background, ot_similarity is in shape of [(len(target_ids)+1) x box_unm]

    #             distance_for_ot = -ot_similarity
    #             mu = zs_weight.new_ones(int(len(target_ids)+1))
    #             nu = zs_weight.new_ones(cur_num_regions)
    #             _, pi = self.sinkhorn(mu, nu, distance_for_ot)
    #             rescale_factor, _ = pi.max(dim=1)
    #             pi = pi / rescale_factor.unsqueeze(1)
    #             max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
    #             fg_mask = matched_gt_inds != int(len(target_ids))

    #             tag_gt_cate = target._pos_category_ids
    #             tag_gt_cate.append(len(zs_weight[0])-1)
    #             matched_gt_inds = [tag_gt_cate[i] for i in matched_gt_inds]     # continugious id --> cat_id
    #             # target_for_ot = torch.zeros_like(similarity_for_img)
    #             y_s = torch.tensor(range(0, cur_num_regions)).to(zs_weight.device)
    #             # target_for_ot = torch.zeros_like(similarity_for_img)
    #             # target_for_ot[matched_gt_inds, y_s] = 1.

    #         neg_concept_list = self._get_negative_concepts(ii, targets, zs_weight.shape[1])
    #         neg_txt_emb = zs_weight[:, neg_concept_list]

    #         # len(matched_gt_inds) == box_num, every box will be matched with a concept
    #         for xx_s, yy_s in zip(matched_gt_inds, y_s):
    #             match_num = 0
    #             # word_features = torch.zeros_like(zs_weight)
    #             # proposal_feature = torch.zeros_like(proj_r[:, 0])
    #             word_features = zs_weight.new_zeros(zs_weight.shape[0], 1+len(neg_concept_list))
    #             proposal_feature = proj_r[:, yy_s].clone()
    #             word_features[:, match_num] = zs_weight[:, xx_s]
    #             match_num = match_num + 1
    #             word_features[:, match_num:len(neg_concept_list)+match_num] = neg_txt_emb
    #             match_num = match_num + len(neg_concept_list)

    #             optimize_matrix = word_features[:,:match_num].permute(1,0) @ proposal_feature / temperature     # (match_num,)
    #             target_matrix = torch.zeros(match_num).to(zs_weight.device)
    #             target_matrix[0] = 1

    #             # import ipdb
    #             # ipdb.set_trace()
    #             # loss += torch.sum(F.binary_cross_entropy_with_logits(optimize_matrix, target_matrix, reduction='none'))
    #             loss += F.cross_entropy(optimize_matrix, target_matrix, reduction='sum')

    #     return loss / batch_size

    # def _get_contrastive_matched_index(self, proj_region_batch, zs_weight, targets, temperature=0.01):
    #     """
    #     Input:
    #         proj_region_batch [List]: [(box_num + 1) x dim] 
    #         zs_weight: dim x cats_num
    #     Output:
    #         matched_indices [List]: [[cat_ids, box_ids], ...]
    #     """
    #     # import ipdb
    #     # ipdb.set_trace()
    #     matched_indices = []
    #     for ii, (proj_r, target) in enumerate(zip(proj_region_batch, targets)):
    #         if self.ignore_zero_region and proj_r.shape[0] <= 1:
    #             # some image from cc3m may be broken and no proposals (excluding the image box) for that
    #             matched_indices.append([[], []])
    #         else:
    #             proj_r = proj_r.permute(1,0)     # dim x box_num
    #             target_ids = target._pos_category_ids 

    #             similarity_for_img = torch.mm(zs_weight.permute(1,0), proj_r) / temperature  # torch.Size([4765, 32]), all_cats x num_box
    #             ss = similarity_for_img[:-1,:]      # remove the last one (for background)
    #             norm_similarity_for_img = ss - ss.min()
    #             ot_similarity = torch.zeros(int(len(target_ids)), proj_r.size()[1]).to(zs_weight.device)
    #             for i, ind in enumerate(target_ids):
    #                 ot_similarity[i] = norm_similarity_for_img[ind]     # get all box scores for each target id, no background id
    #             distance_for_ot = -ot_similarity    # target_id_num x num_box
    #             x_s, y_s = linear_sum_assignment(distance_for_ot.detach().cpu())    # y_s: box id
    #             x_s = [target_ids[i] for i in x_s]    # continugious id --> cat_id

    #             matched_indices.append([x_s, y_s])
    #     return matched_indices

    # def _get_sinkhorn_matched_index(self, proj_region_batch, zs_weight, targets, temperature=0.01):
    #     """
    #     Input:
    #         proj_region_batch [List]: [(box_num + 1) x dim] 
    #         zs_weight: dim x cats_num
    #     Output:
    #         matched_indices [List]: [[cat_ids, box_ids], ...]
    #     """
    #     # import ipdb
    #     # ipdb.set_trace()
    #     matched_indices = []
    #     for ii, (proj_r, target) in enumerate(zip(proj_region_batch, targets)):
    #         if self.ignore_zero_region and proj_r.shape[0] <= 0:
    #             # some image from cc3m may be broken and no proposals (excluding the image box) for that
    #             matched_indices.append([[], []])
    #         else:
    #             proj_r = proj_r.permute(1,0)     # dim x box_num
    #             cur_num_regions = proj_r.shape[1]
    #             target_ids = target._pos_category_ids

    #             similarity_for_img = torch.mm(zs_weight.permute(1,0), proj_r) / temperature   # [4765 x box_unm (32)]
    #             ot_similarity = torch.zeros(int(len(target_ids))+1, proj_r.size()[1]).to(zs_weight.device)
    #             for i, ind in enumerate(target_ids):
    #                 ot_similarity[i] = similarity_for_img[ind]  # fetch scores for each target_id
    #             ot_similarity[-1]  = similarity_for_img[-1]     # background, ot_similarity is in shape of [(len(target_ids)+1) x box_unm]

    #             distance_for_ot = -ot_similarity
    #             mu = zs_weight.new_ones(int(len(target_ids)+1))
    #             nu = zs_weight.new_ones(cur_num_regions)
    #             _, pi = self.sinkhorn(mu, nu, distance_for_ot)
    #             rescale_factor, _ = pi.max(dim=1)
    #             pi = pi / rescale_factor.unsqueeze(1)
    #             max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
    #             fg_mask = matched_gt_inds != int(len(target_ids))

    #             tag_gt_cate = target._pos_category_ids
    #             tag_gt_cate.append(len(zs_weight[0])-1)
    #             matched_gt_inds = [tag_gt_cate[i] for i in matched_gt_inds]     # continugious id --> cat_id
    #             y_s = torch.tensor(range(0, cur_num_regions)).to(zs_weight.device)
    #             matched_indices.append([matched_gt_inds, y_s])
    #     return matched_indices

    # # both ema_contrastive and ema_sinkhorn
    # def ema_align_loss(self, proj_region, proj_region_m, proposals, targets, normalize=True, align_type='ema_contrastive'):
    #     """
    #     Inputs:
    #         proj_region: box_num x dim, no global box region
    #         proj_region_m: box_num x dim, no global box region, from teacher model
    #         openset_txt_em: open_set_num x dim
    #         temperature: 0.01 by default as CLIP
    #     """
    #     # normalize
    #     zs_weight = self.open_txt_emb    # from cfg.MODEL.CLIP.CONCEPT_POOL_EMB
    #     temperature = self.matching_temp
    #     if normalize:
    #         proj_region = F.normalize(proj_region, p=2.0, dim=1)     #  box_num x dim
    #         proj_region_m = F.normalize(proj_region_m, p=2.0, dim=1)     #  box_num x dim
    #         zs_weight = F.normalize(zs_weight, p=2.0, dim=1).permute(1,0)     # dim x open_set_num

    #     batch_size = len(proposals)
    #     num_inst_per_image = [len(p) for p in proposals]
    #     proj_region = proj_region.split(num_inst_per_image, dim=0)     
    #     loss = proj_region[0].new_zeros([1])[0] #(proj_region[0][:,0]).repeat((zs_weight.size()[1]-1,1))

    #     proj_region_m = proj_region_m.split(num_inst_per_image, dim=0)     

    #     with torch.no_grad():
    #         if align_type == 'ema_contrastive':
    #             matched_indices = self._get_contrastive_matched_index(proj_region_m, zs_weight, targets, temperature)
    #         elif align_type == 'ema_sinkhorn':
    #             matched_indices = self._get_sinkhorn_matched_index(proj_region_m, zs_weight, targets, temperature)
    #         else:
    #             raise NotImplementedError

    #     # import ipdb
    #     # ipdb.set_trace()
    #     for ii, (proj_r, target) in enumerate(zip(proj_region, targets)):
    #         proj_r = proj_r.permute(1,0)     #  dim x box_num

    #         neg_concept_list = self._get_negative_concepts(ii, targets, zs_weight.shape[1])
    #         neg_txt_emb = zs_weight[:, neg_concept_list]

    #         x_s, y_s = matched_indices[ii]
    #         for xx_s, yy_s in zip(x_s, y_s):
    #             match_num = 0
    #             word_features = zs_weight.new_zeros(zs_weight.shape[0], 1+len(neg_concept_list))
    #             proposal_feature = proj_r[:, yy_s].clone()
    #             word_features[:, match_num] = zs_weight[:, xx_s]
    #             match_num = match_num + 1
    #             word_features[:, match_num:len(neg_concept_list)+match_num] = neg_txt_emb
    #             match_num = match_num + len(neg_concept_list)
    #             optimize_matrix = word_features[:,:match_num].permute(1,0) @ proposal_feature / temperature     # (match_num,)
    #             target_matrix = torch.zeros(match_num).to(zs_weight.device)
    #             target_matrix[0] = 1

    #             loss += F.cross_entropy(optimize_matrix, target_matrix, reduction='sum')

    #     return loss / batch_size

    # #//TODO: may be a bug when runing on multiple GPUs
    # def weak_alignment_loss(self, all_proj_region, all_proposals, all_targets, normalize=True):
    #     """Inputs:
    #         all_proj_region: List of proj_region. Each proj_region with shape of box_num_in_batch x dim
    #         all_proposals: List of proposals list. 
    #         all_targets: List of targets for the global mini-batch, used for negative samples
    #     """
    #     txt_emb = self.open_txt_emb
    #     # get data for the local batch
    #     # proj_region = all_proj_region[0]
    #     # proposals = all_proposals[0]
    #     # targets = all_targets[0]
    #     local_batch_size = len(all_proposals[0])

    #     # if normalize:
    #     #     proj_region = F.normalize(proj_region, p=2.0, dim=1)     # box_num x dim
    #     #     txt_emb = F.normalize(txt_emb, p=2.0, dim=1)     # open_set_num x dim

    #     # import ipdb
    #     # ipdb.set_trace()
    #     ## create text embeddings
    #     all_targets_list = list(itertools.chain(*all_targets))
    #     all_pos_category_ids = [x._pos_category_ids if hasattr(x, '_pos_category_ids') else [] for x in all_targets_list]
    #     max_cat_num = max([len(x) for x in all_pos_category_ids])

    #     tokens_emb_all = []
    #     noun_masks_all = []
    #     for pos_cat_ids in all_pos_category_ids:
    #         # only consider targets with non-empty _pos_category_ids. This will reduce the actual batch size when box data are used
    #         if len(pos_cat_ids) > 0:
    #             cur_txt_emb = txt_emb[pos_cat_ids, :]   # cat_num x dim

    #             # create noun mask
    #             cur_noun_mask = cur_txt_emb.new_zeros(max_cat_num, dtype=torch.bool)
    #             cur_noun_mask[:len(pos_cat_ids)] = True     # False: should be ignored
    #             noun_masks_all.append(cur_noun_mask)

    #             # padding to max_cat_num
    #             cur_txt_emb = torch.cat([cur_txt_emb, cur_txt_emb.new_zeros(max_cat_num - cur_txt_emb.shape[0], cur_txt_emb.shape[1])], dim=0)  # max_cat_num x dim
    #             tokens_emb_all.append(cur_txt_emb)

    #     noun_masks_all = torch.stack(noun_masks_all, dim=0)     # valid_cap_num x max_cat_num
    #     tokens_emb_all = torch.stack(tokens_emb_all, dim=0)     # valid_cap_num x max_cat_num x dim

    #     ## create region embeddings
    #     temp_box_feats_all = []
    #     for cur_proj_region, cur_proposals, cur_targets in zip(all_proj_region, all_proposals, all_targets):
    #         proj_region_list = cur_proj_region.split([len(x) for x in cur_proposals], dim=0)
    #         for tarIdx, this_target in enumerate(cur_targets):
    #             # only consider targets with non-empty _pos_category_ids. This will reduce the actual batch size when box data are used
    #             if hasattr(this_target, '_pos_category_ids') and len(this_target._pos_category_ids) > 0:
    #                 temp_box_feats_all.append(proj_region_list[tarIdx]) # [box_num x dim]

    #     box_feats_all = []
    #     box_masks_all = []
    #     max_box_num = max([x.shape[0] for x in temp_box_feats_all])
    #     for iidx, box_feats in enumerate(temp_box_feats_all):
    #         box_feats = box_feats.to(txt_emb.device)    # ensure using the same GPU
    #         # create box mask
    #         cur_box_mask = box_feats.new_zeros(max_box_num, dtype=torch.bool)
    #         cur_box_mask[:box_feats.shape[0]] = True    # False: should be ignored
    #         box_masks_all.append(cur_box_mask)

    #         # padding to max_box_num
    #         box_feats = torch.cat([box_feats, box_feats.new_zeros(max_box_num - box_feats.shape[0], box_feats.shape[1])], dim=0)  # max_box_num x dim
    #         box_feats_all.append(box_feats)
        
    #     box_masks_all = torch.stack(box_masks_all, dim=0)     # valid_img_num x max_box_num
    #     box_feats_all = torch.stack(box_feats_all, dim=0)     # valid_img_num x max_box_num x dim

    #     ## get loss value
    #     loss_i2t = self._weak_alignment_loss(box_feats_all[:local_batch_size], box_masks=box_masks_all[:local_batch_size], tokens_emb=tokens_emb_all, noun_masks=noun_masks_all, softmax_dim=1, temperature=self.matching_temp, normalize=normalize)
    #     loss_t2i = self._weak_alignment_loss(box_feats_all, box_masks=box_masks_all, tokens_emb=tokens_emb_all[:local_batch_size], noun_masks=noun_masks_all[:local_batch_size], softmax_dim=0, temperature=self.matching_temp, normalize=normalize)
        
    #     assert torch.isnan(loss_i2t).any() == False, 'nan in loss_i2t, local_batch_size: {}, box_feats_all: {}, tokens_emb_all: {}'.format(local_batch_size, box_feats_all.shape, tokens_emb_all.shape)
    #     assert torch.isnan(loss_t2i).any() == False, 'nan in loss_i2t, local_batch_size: {}, box_feats_all: {}, tokens_emb_all: {}'.format(local_batch_size, box_feats_all.shape, tokens_emb_all.shape)
    #     loss_ita = (loss_i2t + loss_t2i) / 2    # image text alignment

    #     return loss_ita

    # def _weak_alignment_loss(self, box_feats, tokens_emb, box_masks=None, noun_masks=None, reduce_weight=None, softmax_dim=0, temperature=1.0, normalize=True):
        # """
        # args:
        #     box_feats: img_num x box_num x C
        #     tokens_emb: cap_num x token_num x C
        #     box_masks: img_num x box_num
        #     noun_masks: cap_num x token_num
        # """
        # if normalize:
        #     box_feats = F.normalize(box_feats, p=2.0, dim=-1)     # img_num x box_num x C
        #     tokens_emb = F.normalize(tokens_emb, p=2.0, dim=-1)     # img_num x box_num x C

        # # import ipdb
        # # ipdb.set_trace()
        # box_feats_temp = box_feats.unsqueeze(1)     # img_num x 1 x box_num x C
        # tokens_emb_temp = tokens_emb.unsqueeze(0).transpose(-1,-2)   # 1 x cap_num x C x token_num
        # matSimilarity = torch.matmul(box_feats_temp, tokens_emb_temp) / temperature    # img_num x cap_num x box_num x token_num
        
        # # score matrix from similarity matrix
        # if box_masks is not None:
        #     softmax_masks = torch.logical_not(box_masks[:,None,:,None])  # img_num x 1 x box_num x 1
        #     similarity_masked = matSimilarity.masked_fill(softmax_masks, -999)
        #     matScores = similarity_masked.softmax(dim=2)    # img_num x cap_num x box_num x token_num
        # else:
        #     matScores = matSimilarity.softmax(dim=2)    # img_num x cap_num x box_num x token_num 
        
        # matSimilarity_ita = (matScores * matSimilarity).sum(dim=2)  # img_num x cap_num x token_num
        
        # if noun_masks is not None:
        #     noun_masks_temp = noun_masks.unsqueeze(0)  # 1 x cap_num x token_num
        #     matSimilarity_ita = (noun_masks_temp * matSimilarity_ita).sum(dim=2) / (noun_masks_temp.sum(dim=2) + 1e-5)  # img_num x cap_num, //TODO: check matSimilarity_ita always > 0?
        # else:
        #     matSimilarity_ita = matSimilarity_ita.mean(dim=2) # img_num x cap_num

        # sim_targets = torch.zeros(matSimilarity_ita.size()).to(matSimilarity_ita.device)
        # sim_targets.fill_diagonal_(1)

        # if reduce_weight is None:
        #     loss_weakAlign = -torch.sum(F.log_softmax(matSimilarity_ita, dim=softmax_dim)*sim_targets, dim=softmax_dim).mean()
        # else:
        #     # loss_weakAlign = (-torch.sum(torch.log(prob)*sim_targets, dim=softmax_dim) * (p_t ** gamma)).mean()
        #     loss_weakAlign = (-torch.sum(F.log_softmax(matSimilarity_ita, dim=softmax_dim)*sim_targets, dim=softmax_dim) * reduce_weight).mean()
        # return loss_weakAlign

    def region_concept_pseudo_loss(self, region_feats, psuedo_concept_labels=(None, None, None, None, None), use_distill=False, use_contrastive=False):
        """Inputs:
            region_feats: 
            psuedo_concept_labels: concept_scores, target_inds, keep_regions, target_embs, label_mtx
        """
        # # get psuedo concept labels from teacher model
        # concept_scores, target_inds, keep_regions, target_embs, label_mtx \
        #     = self.get_psuedo_concept_labels(images, proposals, gt_instances)
        concept_scores, target_inds, keep_regions, target_embs, label_mtx = psuedo_concept_labels

        # prepare region features for the kept regions
        keep_region_feats = region_feats[keep_regions]
        keep_region_feats = F.normalize(keep_region_feats, p=2, dim=-1)

        loss_region_distill = keep_region_feats.new_zeros([])
        loss_concept_contrastive = keep_region_feats.new_zeros([])

        if use_distill:
            # distillation learning: learns from the predictions of teacher model
            concept_emb = F.normalize(self.open_txt_emb, p=2, dim=-1)    # hanld all-zero vector normalization
            cls_scores = keep_region_feats @ concept_emb.t()  # [#kept_regions, #concepts]
            cls_scores_temp = cls_scores / self.matching_temp
            
            # calculate loss
            loss_region_distill = F.kl_div(F.softmax(cls_scores_temp, dim=1).log(), concept_scores, reduction='batchmean')  # input is log-probabilities, target is probabilities

        if use_contrastive:
            # contrastive learning: matching student visual features with target concept embs
            target_embs = F.normalize(target_embs, p=2, dim=-1)
            match_scores = keep_region_feats @ target_embs.t()  # [#kept_regions, #kept_regions]
            match_scores_temp = match_scores / self.matching_temp

            # calculate loss given matching scores and label matrix
            loss_concept_contrastive = MILCrossEntropy()(match_scores_temp, label_mtx, weights=None, avg_positives=False)

        return loss_region_distill, loss_concept_contrastive

    def image_text_loss(self, img_feats, text_embs, img_feats_allGPU=None, text_embs_allGPU=None, local_batch_size=None):
        """ training with box and captions will block ita loss in backward
            img_feats: image features for this local batch
            text_embs: text embeddings for this local batch
            img_feats_allGPU: should require no gradient
            text_embs_allGPU: should require no gradient
        """
        rank = comm.get_rank()
        word_size = comm.get_world_size()

        # import ipdb
        # ipdb.set_trace()
        if img_feats_allGPU is not None:
            img_feats_allGPU_list = list(img_feats_allGPU.split([local_batch_size]*word_size, dim=0))
            region_feats_full = [img_feats] + img_feats_allGPU_list[:rank] + img_feats_allGPU_list[rank+1:]
            region_feats_full = torch.cat(region_feats_full, dim=0)
        else:
            region_feats_full = img_feats

        if text_embs_allGPU is not None:
            text_embs_allGPU_list = list(text_embs_allGPU.split([local_batch_size]*word_size, dim=0))
            text_embs_full = [text_embs] + text_embs_allGPU_list[:rank] + text_embs_allGPU_list[rank+1:]
            text_embs_full = torch.cat(text_embs_full, dim=0)
        else:
            text_embs_full = text_embs
        
        # remove all-zero embeddings for box data, when both box and captions are used
        if local_batch_size is not None:
            temp_mask = (region_feats_full.abs().sum(dim=1) != 0)
            temp_mask[:local_batch_size] = True # the first local batch must be correct
            region_feats_full = region_feats_full[temp_mask, :]
            text_embs_full = text_embs_full[temp_mask, :]

        # normalize features
        region_feats_full = F.normalize(region_feats_full, p=2, dim=-1)  # img_num_batch x dim
        text_embs_full = F.normalize(text_embs_full, p=2, dim=-1)    # cap_num_batch x dim

        # self.logger.info(f"region_feats_full: {region_feats_full.shape}, text_embs_full: {text_embs_full.shape}, local_batch_size: {local_batch_size}")

        # import ipdb
        # ipdb.set_trace()
        # matching visual features with text embs
        match_scores = region_feats_full @ text_embs_full.view(-1, text_embs_full.size(-1)).t()  # [#regions, img_batch * n_ctx]
        img_b = int(region_feats_full.size(0))
        pooled_score = match_scores

        pooled_score = pooled_score / self.matching_temp
        contrast_target = torch.arange(img_b).to(pooled_score.device)
        row_loss = F.cross_entropy(pooled_score, contrast_target, reduction='none')   # 'none'
        col_loss = F.cross_entropy(pooled_score.t(), contrast_target, reduction='none')

        # only compute losses for this local batch
        row_loss = row_loss[:local_batch_size].mean()
        col_loss = col_loss[:local_batch_size].mean()

        # losses.update({"loss_img_txt_level": (row_loss + col_loss) / 2.0}) 
        loss_img_txt_level = (row_loss + col_loss) / 2.0
        return loss_img_txt_level

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # import ipdb
        # ipdb.set_trace()
        # optional: multiply class scores with RPN scores 
        scores_bf_multiply = scores  # as a backup for visualization purpose
        if self.multiply_rpn_score and not self.training:
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]
        # if self.multiply_rpn_score and not self.training:
        #     if self.rpn_fusion_method == "avg_logits":
        #         # avg two logits, then sigmoid. Better for LVIS trained RPN?
        #         rpn_scores = [p.get('objectness_logits') for p in proposals]
        #         scores = [(inverse_sigmoid(s) + rpn_s[:, None]) / 2 for s, rpn_s in zip(scores, rpn_scores)]
        #         scores = [x.sigmoid() for x in scores]
        #     elif self.rpn_fusion_method == "avg_norm_scores":
        #         # avg two normlaized scores. Better for COCO trained RPN?
        #         rpn_scores = [p.get('objectness_logits') for p in proposals]
        #         rpn_scores = [x.sigmoid() for x in rpn_scores]          # \in [0,1]
        #         scores = [(s + rpn_s[:, None]) / 2 for s, rpn_s in zip(scores, rpn_scores)]
        #     elif self.rpn_fusion_method == "geometric_avg_norm_scores":
        #         rpn_scores = [p.get('objectness_logits') for p in proposals]
        #         rpn_scores = [x.sigmoid() for x in rpn_scores]          # \in [0,1]
        #         alpha = 0.5
        #         scores = [(s ** alpha * rpn_s[:, None] ** (1-alpha)) for s, rpn_s in zip(scores, rpn_scores)]
        #     elif self.rpn_fusion_method == "regionclip":
        #         rpn_scores = [p.get('objectness_logits') for p in proposals]
        #         scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]

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

    def inference_for_pseudo_label(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], threshold=0.9, nms_thres=0.6):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # import ipdb
        # ipdb.set_trace()
        # optional: multiply class scores with RPN scores 
        scores_bf_multiply = scores  # as a backup for visualization purpose
        # if self.multiply_rpn_score:
        #     rpn_scores = [p.get('objectness_logits') for p in proposals]
        #     scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]

        if self.rpn_fusion_method == "avg_logits":
            # avg two logits, then sigmoid. Better for LVIS trained RPN?
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(inverse_sigmoid(s) + rpn_s[:, None]) / 2 for s, rpn_s in zip(scores, rpn_scores)]
            scores = [x.sigmoid() for x in scores]
        elif self.rpn_fusion_method == "avg_norm_scores":
            # avg two normlaized scores. Better for COCO trained RPN?
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            rpn_scores = [x.sigmoid() for x in rpn_scores]          # \in [0,1]
            scores = [(s + rpn_s[:, None]) / 2 for s, rpn_s in zip(scores, rpn_scores)]
        elif self.rpn_fusion_method == "geometric_avg_norm_scores":
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            rpn_scores = [x.sigmoid() for x in rpn_scores]          # \in [0,1]
            alpha = 0.5
            scores = [(s ** alpha * rpn_s[:, None] ** (1-alpha)) for s, rpn_s in zip(scores, rpn_scores)]
        elif self.rpn_fusion_method == "regionclip":
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]

        # import ipdb
        # ipdb.set_trace()
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            threshold,
            nms_thres,
            False,  # disable soft_nms
            self.soft_nms_method,
            self.soft_nms_sigma,
            self.soft_nms_prune,
            self.test_topk_per_image,
            scores_bf_multiply = scores_bf_multiply,
            vis = False,    # disable visualization
        )

    @torch.no_grad()
    def inference_ensemble(
        self, 
        predictions: Tuple[torch.Tensor, torch.Tensor],     # current model outputs, (scores, proposal_deltas)
        predictions_m: Tuple[torch.Tensor, torch.Tensor],    # teacher model outputs, (scores, proposal_deltas)
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
        # import ipdb
        # ipdb.set_trace()
        # scores from current model
        boxes_student = self.predict_boxes(predictions, proposals)
        scores_student = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # scores from teacher model
        # boxes_teacher = self.predict_boxes(predictions_m, proposals)    # theacher.predict_boxes() the same as student's
        scores_teacher = self.predict_probs(predictions_m, proposals)   # theacher.predict_probs() the same as student's

        # score fusion
        scores = []
        for cur_scores_stud, cur_scores_teac in zip(scores_student, scores_teacher):
            cur_scores = cur_scores_stud**(1 - self.ensemble_alpha) * cur_scores_teac**self.ensemble_alpha
            scores.append(cur_scores)

        # avg boxes
        # boxes = [x**(1 - self.ensemble_alpha) * y**self.ensemble_alpha for x, y in zip(boxes_student, boxes_teacher)]
        boxes = boxes_student

        # optional: multiply class scores with RPN scores 
        scores_bf_multiply = scores  # as a backup for visualization purpose
        if self.multiply_rpn_score and not self.training:
            # will be used on LVIS
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]
        
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

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        # import ipdb
        # ipdb.set_trace()
        # don't apply box delta, such as GT boxes
        if self.no_box_delta:
            predict_boxes = proposal_boxes
        # apply box delta
        else:
            predict_boxes = self.box2box_transform.apply_deltas(
                proposal_deltas,
                proposal_boxes,
            )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

