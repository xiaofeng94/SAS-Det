# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) NEC Laboratories America, Inc.
import logging
import copy
from typing import Dict, List, Optional, Tuple
import os
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from numpy.lib import pad
from random import randint
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, InterpolationMode
from torchvision.transforms import functional as tvt_F
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes, PolygonMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
# from detectron2.modeling.backbone.clip_backbone import build_clip_language_encoder, get_clip_tokenzier
# from detectron2.utils.comm import gather_tensors, MILCrossEntropy
from detectron2.utils import comm

from ..backbone.clip_backbone import build_clip_language_encoder, get_clip_tokenzier
# from vldet.modeling import SinkhornDistance

# __all__ = ["CLIPFastRCNN", "PretrainFastRCNN"]



@META_ARCH_REGISTRY.register()
class CLIPFastRCNN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components: 
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        language_encoder: nn.Module, 
        roi_heads: nn.Module,
        # ovd_teacher_backbone: nn.Module,
        # ovd_teacher_roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
        #
        with_image_labels = False,
        with_pseudo_labels = False,
        eval_pseudo_labels = False,
        pl_threshold = 0.9,
        pl_nms_thres = 0.6,
        ema_momentum = -1.0,  # if < 0, ema update not used
        base_cat_ids = None,
        use_adaptive_thres = False,
        min_avg_pls = 1,
        max_avg_pls = 3,
        adaptive_thres_delta = 0.05,
        use_ensemble_eval = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool

        ## added by zsy
        self.with_image_labels = with_image_labels
        self.with_pseudo_labels = with_pseudo_labels
        self.eval_pseudo_labels = eval_pseudo_labels
        self.use_ensemble_eval = use_ensemble_eval

        # self.pl_threshold = pl_threshold
        self.init_pl_thres = pl_threshold   # record init pl_threshold for periodic update
        self.register_buffer("pl_threshold", pl_threshold*torch.ones([]), False)
        # self.pl_threshold = pl_threshold*torch.ones([]) # buffer will be broadcasted if broadcast_buffers == True

        self.use_adaptive_thres = use_adaptive_thres
        self.pl_nms_thres = pl_nms_thres
        self.base_cat_ids = set(base_cat_ids) if base_cat_ids is not None else None

        ## teacher models
        # # will be created the first they're used
        # self.ovd_teacher_backbone = None
        # self.ovd_teacher_roi_heads = None
        self.momentum = ema_momentum
        
        self.with_ovd_teacher = self.with_pseudo_labels or self.use_ensemble_eval
        self.model_pairs = []
        if self.with_ovd_teacher:
            self._create_ovd_teacher()  # teacher backbone is not initialized yet
            self.is_teacher_init = False
        # if self.with_pseudo_labels:
        #     self.ovd_teacher_backbone = copy.deepcopy(backbone)
        #     self.ovd_teacher_roi_heads = copy.deepcopy(roi_heads)
        #     # freeze visual encoder of teacher model
        #     if self.ovd_teacher_backbone is not None:
        #         for p in self.ovd_teacher_backbone.parameters(): 
        #             p.requires_grad = False
        #     if self.ovd_teacher_roi_heads is not None:
        #         for p in self.ovd_teacher_roi_heads.parameters(): 
        #             p.requires_grad = False

        if self.use_adaptive_thres:
            self.PLs_count = 0
            self.PLs_inst_num = 0
            self.PLs_avg_inst_num = 0

            self.min_avg_pls = min_avg_pls
            self.max_avg_pls = max_avg_pls
            self.adaptive_thres_delta = adaptive_thres_delta

        # if self.eval_pseudo_labels:
        #     self.PLs_count = 0
        #     self.PLs_inst_num = 0
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)

    @classmethod
    def from_config(cls, cfg):
        # create independent backbone & RPN
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": 
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
            if cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg)
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        # region proposals are ground-truth boxes
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        
        backbone = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB: # extract concept embeddings
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        
        # get continuous novel cat ids from the external file
        category_info_path = cfg.MODEL.OVD.CATEGORY_INFO
        if (category_info_path is not None) and os.path.exists(category_info_path):
            cat_info = json.load(open(category_info_path, "r"))
            base_cat_ids = cat_info["base_cat_ids"]
        else:
            logging.getLogger(__name__).warning('`MODEL.OVD.CATEGORY_INFO` not exists or None')
            base_cat_ids = None

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
            "language_encoder": language_encoder, 
            "roi_heads": roi_heads, 
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME in ["build_clip_resnet_backbone", "build_clip_resnet_backbone_from_pretrain"],
            # "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "use_clip_attpool": cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            #
            "with_image_labels": cfg.WITH_IMAGE_LABELS,
            "with_pseudo_labels": cfg.MODEL.OVD.WITH_PSEUDO_LABELS,
            "eval_pseudo_labels": cfg.MODEL.OVD.EVAL_PSEUDO_LABELS,
            "pl_threshold": cfg.MODEL.OVD.PL_THRESHOLD,
            "pl_nms_thres": cfg.MODEL.OVD.PL_NMS_THRES,
            'ema_momentum': cfg.MODEL.OVD.EMA_MOMENTUM,
            "base_cat_ids": base_cat_ids,
            #
            "use_adaptive_thres": cfg.MODEL.OVD.USE_ADAPTIVE_THRES,
            "min_avg_pls": cfg.MODEL.OVD.MIN_AVG_PLS,
            "max_avg_pls": cfg.MODEL.OVD.MAX_AVG_PLS,
            "adaptive_thres_delta": cfg.MODEL.OVD.ADAPTIVE_THRES_DELTA,
            #
            "use_ensemble_eval": cfg.MODEL.OVD.USE_ENSEMBLE_EVAL,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _create_ovd_teacher(self):
        self.ovd_teacher_backbone = copy.deepcopy(self.backbone)
        self.ovd_teacher_roi_heads = copy.deepcopy(self.roi_heads)
        # freeze visual encoder of teacher model
        if self.ovd_teacher_backbone is not None:
            for p in self.ovd_teacher_backbone.parameters(): 
                p.requires_grad = False
        if self.ovd_teacher_roi_heads is not None:
            for p in self.ovd_teacher_roi_heads.parameters(): 
                p.requires_grad = False

        # ema pairs, [model, ema_model]
        self.model_pairs = [
            [self.backbone, self.ovd_teacher_backbone],
            [self.roi_heads, self.ovd_teacher_roi_heads],
        ]

    def load_state_dict(self, state_dict, strict: bool = True):    
        # keep running_mean/var whose keys are not in the state_dict. Otherwise, load_state_dict() will reset them to 0/1 for the first time
        # only consider running_mean/var in `roi_heads.text_box_head`. Otherwise, the key `ovd_teacher_backbone` will be added, bugs added for `eval PLs exp`
        model_state_dict = self.state_dict()
        all_model_keys = model_state_dict.keys()
        unmatched_keys = [key for key in all_model_keys if key not in state_dict]
        unmatched_keys_running_mean_var = [key for key in unmatched_keys if (key.startswith('roi_heads.text_box_head') and ('running_mean' in key or 'running_var' in key))]

        preserved_state_dict = {}
        for key in unmatched_keys_running_mean_var:
            preserved_state_dict[key] = model_state_dict[key]
        state_dict.update(preserved_state_dict)

        # import ipdb
        # ipdb.set_trace()
        rets = super().load_state_dict(state_dict=state_dict, strict=strict)

        if self.with_ovd_teacher:
            # running_mean and running_var may be different
            # we need to create teacher model, if no weights for teacher model in the ckpt
            has_ovd_teacher = any(['ovd_teacher_backbone' in x for x in state_dict.keys()])
            has_ovd_teacher = has_ovd_teacher and any(['ovd_teacher_roi_heads' in x for x in state_dict.keys()])

            # may be load multiple times, e.g. CLIP backbone, and offline RPN. _create_ovd_teacher() should be run only once in load_state_dict()
            self.is_teacher_init = self.is_teacher_init or has_ovd_teacher # if has_ovd_teacher, teacher is init by super().load_state_dict()
            if not self.is_teacher_init:
                self._create_ovd_teacher()
                self.is_teacher_init = True
        return rets

    # def remove_base_preds(self, pl_instances, base_cat_ids=None):
    #     assert base_cat_ids is not None, 'base_cat_ids cannot be None for now'
    #     # if base_cat_ids is not None:
    #     for pl_idx, each_instances in enumerate(pl_instances):
    #         keeps = []
    #         pred_classes = each_instances.pred_classes
    #         for _, each_class in enumerate(pred_classes):
    #             if each_class.item() in base_cat_ids:
    #                 keeps.append(False) # ignore PLs for Base
    #             else:
    #                 keeps.append(True)
    #         # if any(keeps):
    #             # import ipdb
    #             # ipdb.set_trace()
    #         # keeps = torch.Tensor(keeps, dtype=torch.bool, device=pred_classes.device)
    #         pl_instances[pl_idx] = each_instances[keeps]
    #     # else:
    #     #     # assume cat id for PL >= self.num_classes (num of base cats)
    #     #     for pl_idx, each_instances in enumerate(pl_instances):
    #     #         keeps = each_instances.pred_classes >= self.roi_heads.num_classes
    #     #         # if any(keeps):
    #     #             # import ipdb
    #     #             # ipdb.set_trace()
    #     #         pl_instances[pl_idx] = each_instances[keeps]
    #     return pl_instances

    def add_ovd_PLs_from_teacher(self, images, proposals, gt_instances=None, ann_type='box', base_cat_ids=None):
        """Assume if ann_type == 'box', data for base categories are available
        """
        # import ipdb
        # ipdb.set_trace()
        features_m = self.ovd_teacher_backbone(images.tensor)
        pl_instances = self.ovd_teacher_roi_heads.get_ovd_pseudo_labels(features_m, proposals, res5=self.ovd_teacher_backbone.layer4, attnpool=self.ovd_teacher_backbone.attnpool, threshold=self.pl_threshold.item(), nms_thres=self.pl_nms_thres)

        # import ipdb
        # ipdb.set_trace()
        if ann_type in ['box']:
            # select results for novel categories
            if base_cat_ids is not None:
                for pl_idx, each_instances in enumerate(pl_instances):
                    keeps = []
                    pred_classes = each_instances.pred_classes
                    for _, each_class in enumerate(pred_classes):
                        if each_class.item() in base_cat_ids:
                            keeps.append(False) # ignore PLs for Base
                        else:
                            keeps.append(True)
                    # if any(keeps):
                        # import ipdb
                        # ipdb.set_trace()
                    # keeps = torch.Tensor(keeps, dtype=torch.bool, device=pred_classes.device)
                    pl_instances[pl_idx] = each_instances[keeps]
            else:
                # assume cat id for PL >= self.num_classes (num of base cats)
                for pl_idx, each_instances in enumerate(pl_instances):
                    keeps = each_instances.pred_classes >= self.roi_heads.num_classes
                    # if any(keeps):
                        # import ipdb
                        # ipdb.set_trace()
                    pl_instances[pl_idx] = each_instances[keeps]

        #     # //TODO: one box is only mapped to one category. may use class-agnostic NMS
        #     pass
        # elif ann_type in ['caption']:
        #     # //TODO: generate pseudo labels for all nouns
        #     pass

        if self.training and self.use_adaptive_thres:
            # avg num of PLs
            self.PLs_inst_num += sum([len(x) for x in pl_instances])
            self.PLs_count += len(pl_instances)
            self.PLs_avg_inst_num = self.PLs_inst_num / (self.PLs_count + 1e-5)

            # update pl_threshold per 200 images
            if self.PLs_count >= 200:
                # avg num of PLs < 1, lower self.pl_threshold
                if self.PLs_avg_inst_num <= self.min_avg_pls:
                    # import ipdb
                    # ipdb.set_trace()
                    # self.pl_threshold = max(self.pl_threshold - 0.05, 0)
                    self.pl_threshold = torch.clamp(self.pl_threshold - self.adaptive_thres_delta, min=0., max=1.)

                # avg num of PLs > 3, increase self.pl_threshold
                if self.PLs_avg_inst_num >= self.max_avg_pls:
                    # import ipdb
                    # ipdb.set_trace()
                    # self.pl_threshold = min(self.pl_threshold + 0.05, 1)
                    self.pl_threshold = torch.clamp(self.pl_threshold + self.adaptive_thres_delta, min=0., max=1.)
                
                # reset
                self.PLs_inst_num = 0
                self.PLs_count = 0
 
        # import ipdb
        # ipdb.set_trace()
        if gt_instances is not None:
            for pl_idx, each_instances in enumerate(pl_instances):
                pl_instances[pl_idx].gt_boxes = each_instances.pred_boxes
                pl_instances[pl_idx].gt_classes = each_instances.pred_classes
                pl_instances[pl_idx].gt_confidence = each_instances.scores
                # get mask place holder for pl_instances, will not used in the training
                if gt_instances[pl_idx].has('gt_masks'):
                    pl_instances[pl_idx].gt_masks = PolygonMasks([[]] * len(pl_instances[pl_idx]))   # no support for bitmask yet
                    # set gt_use_seg for PLs, used in select_foreground_proposals() 
                    pl_instances[pl_idx].gt_use_seg = each_instances.pred_classes.new_zeros(len(pl_instances[pl_idx]))

            # add all-1 scores to gt 
            for gt_idx, each_instances in enumerate(gt_instances):
                gt_instances[gt_idx].gt_confidence = each_instances.gt_boxes.tensor.new_ones(len(each_instances))    # torch.float32
                # set gt_use_seg for gt
                if gt_instances[gt_idx].has('gt_masks'):
                    gt_instances[gt_idx].gt_use_seg = each_instances.gt_classes.new_ones(len(gt_instances[gt_idx]))

            all_instances = [Instances.cat([x, y]) for x, y in zip(gt_instances, pl_instances)]
        else:
            all_instances = pl_instances
        # import ipdb
        # ipdb.set_trace()
        return all_instances

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        self.roi_heads.prepare_running()

        if not self.training:
            with torch.no_grad():
                return self.inference(batched_inputs)

        ann_type = 'box'
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.with_image_labels:
                for inst, x in zip(gt_instances, batched_inputs):
                    inst._ann_type = x['ann_type']
                    inst._pos_category_ids = x['pos_category_ids']
                ann_types = [x['ann_type'] for x in batched_inputs]
                assert len(set(ann_types)) == 1
                ann_type = ann_types[0]
        else:
            gt_instances = None

        # localization branch: offline modules to get the region proposals
        with torch.no_grad():  
            if self.clip_crop_region_type == "GT":  # from ground-truth
                proposals = []
                for r_i, b_input in enumerate(batched_inputs): 
                    this_gt = copy.deepcopy(b_input["instances"])  # Instance
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                    proposals.append(this_gt)                
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.with_pseudo_labels:
            # # create self.ovd_teacher_backbone and self.ovd_teacher_roi_heads the first they're used
            # if self.ovd_teacher_backbone is None or self.ovd_teacher_roi_heads is None:
            #     self._create_ovd_teacher()
            # May be set to True in training script
            if self.ovd_teacher_backbone.training or self.ovd_teacher_roi_heads.training:
                self.ovd_teacher_backbone.eval() 
                self.ovd_teacher_roi_heads.eval()  

            # # import ipdb
            # # ipdb.set_trace()
            # if self.momentum > 0:
            #     self._momentum_update() # update the ema model 

            with torch.no_grad():  
                gt_instances = self.add_ovd_PLs_from_teacher(images, proposals, gt_instances=gt_instances, ann_type=ann_type, base_cat_ids=self.base_cat_ids)

        # import ipdb
        # ipdb.set_trace()
        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, ann_type=ann_type)
            else: # use mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, ann_type=ann_type)
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, attnpool=self.backbone.bottom_up.attnpool)
            else: # use mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        #visualize_proposals(batched_inputs, proposals, self.input_format)

        # record self.PLs_avg_inst_num during training
        if self.with_pseudo_labels and self.use_adaptive_thres:
            storage = get_event_storage()
            storage.put_scalar("z_pseudo_labels/avg_PLs_per_img", self.PLs_avg_inst_num)
            storage.put_scalar("z_pseudo_labels/pl_threshold", self.pl_threshold.item())

            # self.logger.info('[%d] PLs_inst_num: %d, PLs_count: %d, avg_PLs_per_img: %.02f, pl_threshold: %.02f' % (comm.get_rank(), self.PLs_inst_num, self.PLs_count, self.PLs_avg_inst_num, self.pl_threshold.item()))
            # detector_losses.update({"z_avg_PLs_per_img": images.tensor.new_ones([])*self.PLs_avg_inst_num})

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        
        # localization branch: offline modules to get the region proposals
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes} #, 'objectness_logits': None}
                proposals.append(this_gt)                
        elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
    
        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        
        # ## for debugging
        # features = self.teacher_backbone(images.tensor)
        # # Given the proposals, crop region features from 2D image features and classify the regions
        # if self.use_clip_c4: # use C4 + resnet weights from CLIP
        #     if self.use_clip_attpool: # use att_pool from CLIP to match dimension
        #         results, _ = self.ovd_teacher_roi_heads(images, features, proposals, None, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)
        #     else: # use mean pool
        #         results, _ = self.ovd_teacher_roi_heads(images, features, proposals, None, res5=self.teacher_backbone.layer4)
        # else:  # regular detector setting
        #     if self.use_clip_attpool: # use att_pool from CLIP to match dimension
        #         results, _  = self.ovd_teacher_roi_heads(images, features, proposals, None, attnpool=self.teacher_backbone.bottom_up.attnpool)
        #     else:
        #         results, _  = self.ovd_teacher_roi_heads(images, features, proposals, None)

        # import ipdb
        # ipdb.set_trace()
        if self.eval_pseudo_labels:
            ## eval pseudo labels and output avg # PLs
            results = self.add_ovd_PLs_from_teacher(images, proposals, base_cat_ids=self.base_cat_ids)
            # self.PLs_count += 1
            # self.PLs_inst_num += len(results[0])

            # if self.PLs_count > int(1975//comm.get_world_size()):
            #     msg = 'avg inst: %.4f (%d images)' % (self.PLs_inst_num/self.PLs_count, self.PLs_count)
            #     print(msg)
            #     self.logger.info(msg)

            # world_size = comm.get_world_size()
            # if world_size > 1:
            #     # image counts from all GPUs
            #     debug_count_allGPU = [torch.zeros([], device=self.device) for _ in range(world_size)]
            #     dist.all_gather(debug_count_allGPU, torch.tensor(self.PLs_count, device=self.device))
            #     cur_count_all = sum([x.item() for x in debug_count_allGPU])
            #     # num of predicted instances from all GPUs
            #     debug_inst_num_allGPU = [torch.zeros([], device=self.device) for _ in range(world_size)]
            #     dist.all_gather(debug_inst_num_allGPU, torch.tensor(self.PLs_inst_num, device=self.device))
            #     cur_inst_all = sum([x.item() for x in debug_inst_num_allGPU])
            # else:
            #     cur_count_all = self.PLs_count
            #     cur_inst_all = self.PLs_inst_num
            # # 4800 for coco ovd
            # if comm.is_main_process() and cur_count_all > 4800:
            #     self.logger.info('avg inst: %.4f (%d images)' % (cur_inst_all/cur_count_all, cur_count_all))
            #     # import ipdb
            #     # ipdb.set_trace()
        elif self.use_ensemble_eval:
            ## ensemble predictions from the current model and the teacher
            features = self.backbone(images.tensor)
            assert self.use_clip_c4
            assert self.use_clip_attpool
            results = self.roi_heads.inference_ensemble(images, features, proposals, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, ema_inputs=[self.ovd_teacher_backbone, self.ovd_teacher_roi_heads])
        else:
            ## original eval
            features = self.backbone(images.tensor)
            # Given the proposals, crop region features from 2D image features and classify the regions
            if self.use_clip_c4: # use C4 + resnet weights from CLIP
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
                else: # use mean pool
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
            else:  # regular detector setting
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    results, _  = self.roi_heads(images, features, proposals, None, attnpool=self.backbone.bottom_up.attnpool)
                else:
                    results, _  = self.roi_heads(images, features, proposals, None)
        
        # # for debug only
        # results = self.remove_base_preds(results, base_cat_ids=self.base_cat_ids)

        #visualize_proposals(batched_inputs, proposals, self.input_format)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFastRCNN._postprocess(results, batched_inputs)
        else:
            return results

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()        
    def periodic_update_pairs(self):
        # set self.pl_threshold to a low value so that more PLs involved at the beginning of each update
        self.pl_threshold[...] = self.init_pl_thres
        self.logger.info('[%d] Periodic update, reset pl_threshold to %.02f' % (comm.get_rank(), self.init_pl_thres))

        # using copy_ to avoid BUGs related to variable by reference
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results








# @META_ARCH_REGISTRY.register()
# class PretrainFastRCNN(nn.Module):
#     """
#     RegionCLIP: Learning visual region representation via vision-language pretraining from image-text pairs
#     1. region-token level matching: learn to match the pseudo region-text pairs, provided by teacher model
#     2. image-text level matching: learn to match image-text pairs, obtained from the Internet
#     """
#     @configurable
#     def __init__(
#         self,
#         *,
#         offline_backbone: Backbone,
#         backbone: Backbone,
#         offline_proposal_generator: nn.Module,
#         roi_heads: nn.Module,
#         teacher_backbone: nn.Module,
#         teacher_roi_heads: nn.Module,
#         pixel_mean: Tuple[float],
#         pixel_std: Tuple[float],
#         input_format: Optional[str] = None,
#         vis_period: int = 0,
#         clip_crop_region_type: str = 'GT',
#         use_clip_c4: False,
#         use_clip_attpool: False,
#         offline_input_format: Optional[str] = None,
#         offline_pixel_mean: Tuple[float],
#         offline_pixel_std: Tuple[float],
#         language_encoder: nn.Module,
#         matching_temp: None,
#         num_regions_per_img: int = 0,
#         img_txt_level: None,
#         gather_gpus: False,
#         concept_emb: None,
#     ):
#         """
#         Args:
#             backbone: a backbone module, must follow detectron2's backbone interface
#             proposal_generator: a module that generates proposals using backbone features
#             roi_heads: a ROI head that performs per-region computation
#             pixel_mean, pixel_std: list or tuple with #channels element, representing
#                 the per-channel mean and std to be used to normalize the input image
#             input_format: describe the meaning of channels of input. Needed by visualization
#             vis_period: the period to run visualization. Set to 0 to disable.
#         """
#         super().__init__()
#         self.offline_backbone = offline_backbone
#         self.backbone = backbone
#         self.offline_proposal_generator = offline_proposal_generator
#         self.roi_heads = roi_heads

#         self.input_format = input_format
#         self.vis_period = vis_period
#         if vis_period > 0:
#             assert input_format is not None, "input_format is required for visualization!"

#         # input format, pixel mean and std for offline modules
#         self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
#         assert (
#             self.pixel_mean.shape == self.pixel_std.shape
#         ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
#         if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
#             assert input_format == 'RGB'
#             self.div_pixel = True
#         else:
#             self.div_pixel = False

#         if offline_input_format and offline_pixel_mean and offline_pixel_std:
#             self.offline_input_format = offline_input_format
#             self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
#             self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
#             if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
#                 assert offline_input_format == 'RGB'
#                 self.offline_div_pixel = True
#             else:
#                 self.offline_div_pixel = False
        
#         self.clip_crop_region_type = clip_crop_region_type
#         self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
#         self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        
#         # image-text level pretraining
#         self.img_txt_level = img_txt_level[0]
#         self.only_eot = img_txt_level[1]
#         if self.img_txt_level:
#             self.lang_encoder = language_encoder
#             for p in self.lang_encoder.parameters():  # freeze language encoder
#                 p.requires_grad = False
#         self.matching_temp = matching_temp
#         self.context_length = 77 # defined in clip_img_txt_pair_tsv class
#         self.num_regions_per_img = num_regions_per_img
#         self.gather_gpus = gather_gpus

#         # region-token level pretraining
#         if concept_emb[0]:
#             self.register_buffer("concept_emb", torch.load(concept_emb[0]), False) # [#concepts, d]
#             self.concept_thres = concept_emb[1]

#             self.teacher_backbone = teacher_backbone
#             # freeze visual encoder of teacher model
#             if self.teacher_backbone is not None:
#                 for p in self.teacher_backbone.parameters(): 
#                     p.requires_grad = False
#             self.teacher_roi_heads = teacher_roi_heads
#             if self.teacher_roi_heads is not None:
#                 for p in self.teacher_roi_heads.parameters(): 
#                     p.requires_grad = False

#             if concept_emb[2] is None: # teacher model uses the same concept embedding as student model
#                 self.register_buffer("teacher_concept_emb", torch.load(concept_emb[0]), False)
#             else: # teacher model uses a seperate concept embedding
#                 self.register_buffer("teacher_concept_emb", torch.load(concept_emb[2]), False)
#         else:
#             self.concept_emb = None

#     @classmethod
#     def from_config(cls, cfg):
#         if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": # create isolated backbone & RPN
#             # create offline cfg for the pretrained backbone & RPN
#             from detectron2.config import get_cfg
#             offline_cfg = get_cfg()
#             offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
#             if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
#                 offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
#                 offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
#                 offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
#                 offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
#             if cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS:
#                 offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS 
#             if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
#                 offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH
            
#             # create offline backbone and RPN
#             offline_backbone = build_backbone(offline_cfg) # build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
#             offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())
#             # convert to evaluation mode
#             for p in offline_backbone.parameters(): p.requires_grad = False
#             for p in offline_rpn.parameters(): p.requires_grad = False
#             offline_backbone.eval()
#             offline_rpn.eval()
#         elif cfg.MODEL.CLIP.CROP_REGION_TYPE in ["GRID", "RANDOM"]:
#             offline_backbone = None
#             offline_rpn = None
#             offline_cfg = None
        
#         # visual encoder and roi_heads of student model
#         backbone = build_backbone(cfg)
#         roi_heads = build_roi_heads(cfg, backbone.output_shape())
#         # language encoder of student model
#         language_encoder = build_clip_language_encoder(cfg)
#         # visual encoder of teacher model
#         teacher_cfg = copy.deepcopy(cfg)
#         teacher_cfg.defrost()
#         teacher_cfg.MODEL.RESNETS.DEPTH = teacher_cfg.MODEL.CLIP.TEACHER_RESNETS_DEPTH
#         teacher_backbone = build_backbone(teacher_cfg)

#         teacher_cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = teacher_cfg.MODEL.CLIP.TEACHER_POOLER_RESOLUTION
#         teacher_roi_heads_name = teacher_cfg.MODEL.CLIP.TEACHER_ROI_HEADS_NAME
#         if teacher_roi_heads_name:
#             teacher_cfg.MODEL.ROI_HEADS.NAME = teacher_roi_heads_name
#         teacher_roi_heads = build_roi_heads(teacher_cfg, teacher_backbone.output_shape())

#         return {
#             "offline_backbone": offline_backbone,
#             "offline_proposal_generator": offline_rpn, 
#             "backbone": backbone,
#             "roi_heads": roi_heads, 
#             "teacher_backbone": teacher_backbone,
#             "teacher_roi_heads": teacher_roi_heads,
#             "input_format": cfg.INPUT.FORMAT,
#             "vis_period": cfg.VIS_PERIOD,
#             "pixel_mean": cfg.MODEL.PIXEL_MEAN,
#             "pixel_std": cfg.MODEL.PIXEL_STD,
#             "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
#             "use_clip_c4": cfg.MODEL.BACKBONE.NAME in ["build_clip_resnet_backbone", "build_clip_resnet_backbone_from_pretrain"],
#             # "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
#             "use_clip_attpool": cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
#             "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
#             "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
#             "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
#             "language_encoder": language_encoder,
#             "matching_temp": cfg.MODEL.CLIP.CLSS_TEMP,
#             "num_regions_per_img": cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS,
#             "img_txt_level": (cfg.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL, cfg.MODEL.CLIP.PRETRAIN_ONLY_EOT),
#             "gather_gpus": cfg.MODEL.CLIP.GATHER_GPUS,
#             "concept_emb": (cfg.MODEL.CLIP.CONCEPT_POOL_EMB, cfg.MODEL.CLIP.CONCEPT_THRES, cfg.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB),
#         }

#     @property
#     def device(self):
#         return self.pixel_mean.device

#     def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:

#                 * image: Tensor, image in (C, H, W) format.
#                 * instances (optional): groundtruth :class:`Instances`
#                 * proposals (optional): :class:`Instances`, precomputed proposals.

#                 Other information that's included in the original dicts, such as:

#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                   See :meth:`postprocess` for details.

#         Returns:
#             list[dict]:
#                 Each dict is the output for one input image.
#                 The dict contains one key "instances" whose value is a :class:`Instances`.
#                 The :class:`Instances` object has the following keys:
#                 "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
#         """
#         if not self.training:
#             return self.inference(batched_inputs)
#         gt_instances = None
#         # import ipdb
#         # ipdb.set_trace()
#         losses = {}
#         # losses['loss_cls'] = batched_inputs[0]['image'].to(self.device).new_zeros([1])[0]

#         # localization branch: offline modules to get the region proposals
#         proposals = self.get_region_proposals(batched_inputs)
#         global_proposals = self.create_global_proposals(batched_inputs)

#         # recognition branch: get 2D feature maps using the backbone of recognition branch and extract region features
#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)
#         region_feats = self.get_region_features(images, features, proposals, gt_instances)
#         global_feats = self.get_region_features(images, features, global_proposals, gt_instances)

#         # image-text level matching
#         if self.img_txt_level:
#             self.image_text_matching(batched_inputs, proposals, region_feats, losses, global_feats=global_feats)

#         # region-concept level matching
#         if self.concept_emb is not None:
#             self.region_concept_matching(images, proposals, gt_instances, region_feats, losses)

#         return losses

#     def region_concept_matching(self, images, proposals, gt_instances, region_feats, losses, use_distill=True, use_contrastive=True):
#         # get psuedo concept labels from teacher model
#         concept_scores, target_inds, keep_regions, target_embs, label_mtx \
#             = self.get_psuedo_concept_labels(images, proposals, gt_instances)

#         # prepare region features for the kept regions
#         keep_region_feats = region_feats[keep_regions]
#         # keep_region_feats = keep_region_feats / keep_region_feats.norm(dim=-1, keepdim=True)
#         keep_region_feats = F.normalize(keep_region_feats, p=2, dim=-1)

#         if use_distill:
#             # distillation learning: learns from the predictions of teacher model
#             # concept_emb = self.concept_emb / self.concept_emb.norm(dim=-1, keepdim=True)
#             concept_emb = F.normalize(self.concept_emb, p=2, dim=-1)    # hanld all-zero vector normalization
#             cls_scores = keep_region_feats @ concept_emb.t()  # [#kept_regions, #concepts]
#             cls_scores_temp = cls_scores / self.matching_temp
            
#             # calculate loss
#             cls_loss = F.kl_div(F.softmax(cls_scores_temp, dim=1).log(), concept_scores, reduction='batchmean')  # input is log-probabilities, target is probabilities
#             losses.update({"loss_region_distill": cls_loss}) #  * 0.8})

#         if use_contrastive:
#             # contrastive learning: matching student visual features with target concept embs
#             # target_embs = target_embs / target_embs.norm(dim=-1, keepdim=True)
#             target_embs = F.normalize(target_embs, p=2, dim=-1)
#             match_scores = keep_region_feats @ target_embs.t()  # [#kept_regions, #kept_regions]
#             match_scores_temp = match_scores / self.matching_temp

#             # calculate loss given matching scores and label matrix
#             contrastive_loss = MILCrossEntropy()(match_scores_temp, label_mtx, weights=None, avg_positives=False)
#             losses.update({"loss_concept_contrastive": contrastive_loss})

#     def image_text_matching(self, batched_inputs, proposals, region_feats, losses, global_feats):
#         # encode text
#         num_cap = int(batched_inputs[0][1].size(0) / self.context_length)
#         if num_cap == 1:  # one caption per image
#             text = [x[1].view(1,-1).to(self.device) for x in batched_inputs]
#         else: # multiple caption pers image, then randomly pick one
#             rand_ind = [randint(0, num_cap-1) for _ in range(len(batched_inputs))]
#             text = [x[1].view(-1,self.context_length)[rand_ind[i]:rand_ind[i]+1].to(self.device) for i, x in enumerate(batched_inputs)]
#         text = torch.cat(text, dim=0)
#         text_embs = self.lang_encoder.encode_text(text, only_eot=self.only_eot)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]

#         # prepare region features and text embeddings
#         region_feats = global_feats
#         # region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)
#         # text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
#         region_feats = F.normalize(region_feats, p=2, dim=-1)
#         text_embs = F.normalize(text_embs, p=2, dim=-1)

#         region_feats_full, min_bs = gather_tensors(region_feats) if self.gather_gpus else (region_feats, None)  #  gather across GPUs
#         text_embs_full, min_bs = gather_tensors(text_embs) if self.gather_gpus else (text_embs, None)  #  gather across GPUs

#         # matching visual features with text embs
#         match_scores = region_feats_full @ text_embs_full.view(-1, text_embs_full.size(-1)).t()  # [#regions, img_batch * n_ctx]
#         img_b = int(region_feats_full.size(0))
#         pooled_score = match_scores

#         pooled_score = pooled_score / self.matching_temp
#         contrast_target = torch.arange(img_b).to(self.device)
#         row_loss = F.cross_entropy(pooled_score, contrast_target)
#         col_loss = F.cross_entropy(pooled_score.t(), contrast_target)
#         losses.update({"loss_img_txt_level": (row_loss + col_loss) / 2.0}) 

#     def get_psuedo_concept_labels(self, images, proposals, gt_instances, s_temp=0.01):
#         """ Input images and region proposals, return matching results from teacher model
#         """
#         # import ipdb
#         # ipdb.set_trace()
#         with torch.no_grad():
#             # extract visual features from teacher model
#             features = self.teacher_backbone(images.tensor)
#             teacher_region_feats = self.teacher_roi_heads(images, features, proposals, gt_instances, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)
            
#             # match teacher visual features with teacher concept embs to create pseudo labels
#             # teacher_region_feats = teacher_region_feats / teacher_region_feats.norm(dim=-1, keepdim=True)
#             # teacher_concept_emb = self.teacher_concept_emb / self.teacher_concept_emb.norm(dim=-1, keepdim=True)
#             teacher_region_feats = F.normalize(teacher_region_feats, p=2, dim=-1)
#             teacher_concept_emb = F.normalize(self.teacher_concept_emb, p=2, dim=-1)

#             concept_scores = teacher_region_feats @ teacher_concept_emb.t()  # [#regions, #concepts]
#             concept_scores = F.softmax(concept_scores / s_temp, dim=1)

#             max_scores, max_inds = torch.max(concept_scores, dim=1)
#             keep_regions = max_scores > self.concept_thres  # only keep the regions that have high matching score with a concept
#             if keep_regions.nonzero().size(0) == 0: # if all regions can't match to any concept
#                 print("all regions can't match to any concept!")
#                 keep_regions = max_scores > 0.0 
#             target_inds = max_inds[keep_regions]
#             target_embs = self.concept_emb[target_inds] # the target embedding of student model
#             label_mtx = (target_inds.view(-1, 1) == target_inds.view(1, -1)).type_as(teacher_region_feats)
#             concept_scores = concept_scores[keep_regions]
            
#         return concept_scores, target_inds, keep_regions, target_embs, label_mtx

#     def get_region_features(self, images, features, proposals, gt_instances):
#         """ Input images and region proposals, return region features
#         """
#         # Given the proposals, crop region features from 2D image features
#         if self.use_clip_c4: # use C4 + resnet weights from CLIP
#             if self.use_clip_attpool: # use att_pool from CLIP to match dimension
#                 region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
#             else: # use mean pool
#                 region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
#         else:  # regular detector setting
#             region_feats = self.roi_heads(images, features, proposals, gt_instances)
#         return region_feats

#     def get_region_proposals(self, batched_inputs):
#         """ Given image, return object proposals
#         """
#         with torch.no_grad():  
#             if self.clip_crop_region_type == "RANDOM":  # from random proposals
#                 proposals = self.create_rand_boxes(batched_inputs)         
#             elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
#                 if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
#                     self.offline_backbone.eval() 
#                     self.offline_proposal_generator.eval()  
#                 images = self.offline_preprocess_image(batched_inputs)
#                 features = self.offline_backbone(images.tensor)
#                 if self.offline_proposal_generator is not None:
#                     proposals, _ = self.offline_proposal_generator(images, features, None)     
#             #visualize_proposals(batched_inputs, proposals, self.input_format, vis_pretrain=True)
        
#         # randomly select proposals
#         if self.training:
#             rand_inds = [torch.randperm(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
#             proposals = [p[rand_inds[i]] for i, p in enumerate(proposals)]
#         return proposals

#     def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
#         Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
#         Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
#         """
#         images = [x[0].to(self.device) for x in batched_inputs]
#         if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
#             (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
#             images = [x[[2,1,0],:,:] for x in images]
#         if self.offline_div_pixel:
#             images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
#         else:
#             images = [((x * 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
#         return images

#     def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
#         Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
#         Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
#         """
#         images = [x[0].to(self.device) for x in batched_inputs]
#         if self.div_pixel:
#             images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#         else:
#             images = [((x * 255.0) - self.pixel_mean) / self.pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images

#     def create_rand_boxes(self, batched_inputs, grid_length=8):
#         """ create random boxes within an image, output random self.num_regions_per_img boxes
#         return a list of Boxes
#         """
#         images = self.preprocess_image(batched_inputs)
#         image_height = images.tensor.size(2)
#         image_width = images.tensor.size(3)

#         left_top_x = torch.tensor([i*(grid_length) for i in range(image_width // grid_length)])
#         left_top_y = torch.tensor([i*(grid_length) for i in range(image_height // grid_length)])
#         right_bot_x = torch.tensor([(i+1)*(grid_length) for i in range(image_width // grid_length)])
#         right_bot_y = torch.tensor([(i+1)*(grid_length) for i in range(image_height // grid_length)])
#         x_inds = torch.randint(0, left_top_x.size(0), (self.num_regions_per_img,))
#         y_inds = torch.randint(0, left_top_y.size(0), (self.num_regions_per_img,))

#         proposals = []
#         for i in range(self.num_regions_per_img):
#             rb_x_candidates = right_bot_x[x_inds[i]:]
#             rb_x = rb_x_candidates[torch.randperm(rb_x_candidates.size(0))[0]]
#             rb_y_candidates = right_bot_y[y_inds[i]:]
#             rb_y = rb_y_candidates[torch.randperm(rb_y_candidates.size(0))[0]]
#             this_box = torch.cat((left_top_x[x_inds[i]].view(1,1), left_top_y[y_inds[i]].view(1,1), rb_x.view(1,1), rb_y.view(1,1)),dim=-1)
#             proposals.append(this_box)
#         proposals = torch.cat(proposals).float().to(self.device)
#         proposals = [Boxes(proposals) for i in range(len(batched_inputs))] # a list of Boxes
#         return proposals

#     def create_global_proposals(self, batched_inputs):
#         """ create a single global box for an image, so as to extract global image features with RoIAlign on high-resolution images.
#         """
#         images = self.preprocess_image(batched_inputs)
#         image_height = images.tensor.size(2)
#         image_width = images.tensor.size(3)

#         global_box = torch.tensor([0, 0, image_width, image_height]).view(1,4).float().to(self.device)
#         # proposals = [Boxes(global_box) for i in range(len(batched_inputs))] # a list of Boxes
#         proposals = []
#         for iidx in range(len(batched_inputs)):
#             curInst = Instances((image_height, image_width))
#             curInst.proposal_boxes = Boxes(global_box)
#             proposals.append(curInst)
#         return proposals

#     def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
#         pass

#     @staticmethod
#     def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Rescale the output instances to the target size.
#         """
#         # note: private function; subject to changes
#         processed_results = []
#         for results_per_image, input_per_image in zip(instances, batched_inputs):
#             height, width = input_per_image[-1][2] # original image size, before resizing
#             r = detector_postprocess(results_per_image, height, width)
#             processed_results.append({"instances": r})
#         return processed_results


# def visualize_proposals(batched_inputs, proposals, input_format, vis_pretrain=False):
#     """
#     A function used to visualize images and proposals. It shows ground truth
#     bounding boxes on the original image and up to 20 top-scoring predicted
#     object proposals on the original image. Users can implement different
#     visualization functions for different models.

#     Args:
#         batched_inputs (list): a list that contains input to the model.
#         proposals (list): a list that contains predicted proposals. Both
#             batched_inputs and proposals should have the same length.
#     """
#     from detectron2.utils.visualizer import Visualizer

#     max_vis_prop = 50
#     if vis_pretrain:
#         for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
#             # img = input[0] * 255.0
#             img = input["image"]
#             img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
#             box_size = min(len(prop.proposal_boxes), max_vis_prop)
#             v_pred = Visualizer(img, None)
#             v_pred = v_pred.overlay_instances(
#                 boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
#             )
#             prop_img = v_pred.get_image()
#             vis_img = prop_img
#             to_save = Image.fromarray(np.array(vis_img, np.uint8))
#             to_save.save("output/regions/" + str(i) + ".png")
#             #break  # only visualize one image in a batch
#     else:
#         for input, prop in zip(batched_inputs, proposals):
#             img = input["image"]
#             img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
#             v_gt = Visualizer(img, None)
#             v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
#             anno_img = v_gt.get_image()
#             box_size = min(len(prop.proposal_boxes), max_vis_prop)
#             v_pred = Visualizer(img, None)
#             v_pred = v_pred.overlay_instances(
#                 boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
#             )
#             prop_img = v_pred.get_image()
#             vis_img = np.concatenate((anno_img, prop_img), axis=1)
#             #vis_img = vis_img.transpose(2, 0, 1)
#             vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
#             f_n = input['file_name']
#             to_save = Image.fromarray(np.array(vis_img, np.uint8))
#             to_save.save("output/regions/" + f_n.split("/")[-1].split(".")[0] + ".png")
#             #break  # only visualize one image in a batch


# @META_ARCH_REGISTRY.register()
# class WeakPretrainFastRCNN(PretrainFastRCNN):
#     """
#     RegionCLIP: Learning visual region representation via vision-language pretraining from image-text pairs
#     1. region-token level matching: learn to match the pseudo region-text pairs, provided by teacher model
#     2. image-text level matching: learn to match image-text pairs, obtained from the Internet
#     """
#     @configurable
#     def __init__(
#         self,
#         *,
#         ignore_zero_region=False,
#         weak_loss_type='contrastive',
#         weak_loss_weight=0.01,
#         image_loss_weight=0.1,
        
#         # box_select_thres=0.9, # moved to roi head
#         neg_concept_num=10,
#         momentum=0.999,
#         ignore_cls_loss=False,
#         # open_txt_emb_path=None,
#         dataset_bs=None,
#         text_emb_dim=1024,
#         random_sample_region=False, # if random pick regions in the topk proposals
#         **kwargs
#     ):
#         """
#         Args:
#             backbone: a backbone module, must follow detectron2's backbone interface
#             proposal_generator: a module that generates proposals using backbone features
#             roi_heads: a ROI head that performs per-region computation
#             pixel_mean, pixel_std: list or tuple with #channels element, representing
#                 the per-channel mean and std to be used to normalize the input image
#             input_format: describe the meaning of channels of input. Needed by visualization
#             vis_period: the period to run visualization. Set to 0 to disable.
#         """
#         # import ipdb
#         # ipdb.set_trace()
#         super().__init__(**kwargs)
#         self.ignore_zero_region = ignore_zero_region
#         self.weak_loss_type = weak_loss_type
#         self.weak_loss_weight = weak_loss_weight
#         self.image_loss_weight = image_loss_weight
#         # self.box_select_thres = box_select_thres
#         self.neg_concept_num = neg_concept_num
#         self.ignore_cls_loss = ignore_cls_loss
        
#         # if 'ema' in self.weak_loss_type:
#         #     self.momentum = momentum
#         #     # ema pairs, [model, ema_model]
#         #     self.model_pairs = [
#         #         [self.backbone, self.teacher_backbone],
#         #     ]

#         # use self.concept_emb
#         self.concept_emb = torch.cat(
#             [self.concept_emb, self.concept_emb.new_zeros((1, self.concept_emb.shape[1]))], dim=0
#         ) # (num + 1) x dim
#         self.teacher_concept_emb = torch.cat(
#             [self.teacher_concept_emb, self.teacher_concept_emb.new_zeros((1, self.teacher_concept_emb.shape[1]))], dim=0
#         ) # (num + 1) x dim
        
#         # if open_txt_emb_path is not None:
#         #     open_txt_emb = torch.load(open_txt_emb_path)
#         #     open_txt_emb = torch.cat(
#         #         [open_txt_emb, open_txt_emb.new_zeros((1, open_txt_emb.shape[1]))], dim=0
#         #     ) # (num + 1) x dim
#         #     self.register_buffer('open_txt_emb', open_txt_emb)  # un-normalized

#         self.tokenizer = get_clip_tokenzier()

#         self.sinkhorn = SinkhornDistance(eps = 1e-3, max_iter=100)
#         # self.clip_transform = get_clip_image_transform(224)
#         self.center_crop = CenterCrop(224)

#         # self.dataset_bs = dataset_bs
#         # # self.det_cap_data_ratio = None if dataset_bs is None else int(dataset_bs[1]/dataset_bs[0])
#         # self.text_emb_dim = text_emb_dim

#         self.random_sample_region = random_sample_region

#         self.logger = logging.getLogger("detectron2.trainer")
#         # self.debug = True
#         # self.debug_count = 0

#     @classmethod
#     def from_config(cls, cfg):
#         ret = super().from_config(cfg)
#         ret.update({
#             'ignore_zero_region': False,
#             'weak_loss_type': cfg.MODEL.WEAK_LOSS.WEAK_LOSS_TYPE,
#             'weak_loss_weight': cfg.MODEL.WEAK_LOSS.WEAK_LOSS_WEIGHT,
#             'image_loss_weight': cfg.MODEL.WEAK_LOSS.IMAGE_LOSS_WEIGHT,
#             # 'open_txt_emb_path': cfg.MODEL.CLIP.OPENSET_TEXT_EMB_PATH,
#             # "teacher_backbone": None,
#             # "teacher_roi_heads": None,
#             # "box_select_thres": cfg.MODEL.WEAK_LOSS.BOX_SELECT_THRES,
#             "neg_concept_num": cfg.MODEL.WEAK_LOSS.NEG_CONCEPT_NUM,
#             "momentum": cfg.MODEL.WEAK_LOSS.MOMENTUM,
#             "ignore_cls_loss": cfg.MODEL.IGNORE_CLS_LOSS,
#             # #
#             # "dataset_bs": cfg.DATALOADER.DATASET_BS,
#             # "text_emb_dim": cfg.MODEL.CLIP.TEXT_EMB_DIM, 
#             "random_sample_region": cfg.MODEL.CLIP.RANDOM_SAMPLE_REGION,
#         })
#         return ret

#     def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:

#                 * image: Tensor, image in (C, H, W) format. \in [0, 255]
#                 * instances (optional): groundtruth :class:`Instances`
#                 * proposals (optional): :class:`Instances`, precomputed proposals.

#                 Other information that's included in the original dicts, such as:

#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                   See :meth:`postprocess` for details.

#         Returns:
#             list[dict]:
#                 Each dict is the output for one input image.
#                 The dict contains one key "instances" whose value is a :class:`Instances`.
#                 The :class:`Instances` object has the following keys:
#                 "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
#         """
#         self.roi_heads.prepare_running()

#         if not self.training:
#             return self.inference(batched_inputs)

#         # ### regionclip loss
#         # gt_instances = None
#         # # import ipdb
#         # # ipdb.set_trace()
#         # losses = {}
#         # # losses['loss_cls'] = batched_inputs[0]['image'].to(self.device).new_zeros([1])[0]

#         # # localization branch: offline modules to get the region proposals
#         # proposals = self.get_region_proposals(batched_inputs, random_pick=True, select_topk=False)
#         # global_proposals = self.create_global_proposals(proposals)

#         # # recognition branch: get 2D feature maps using the backbone of recognition branch and extract region features
#         # images = self.preprocess_image(batched_inputs)
#         # features = self.backbone(images.tensor)
#         # region_feats = self.get_region_features(images, features, proposals, gt_instances)
#         # global_feats = self.get_region_features(images, features, global_proposals, gt_instances)

#         # # image-text level matching
#         # if self.img_txt_level:
#         #     inds = [torch.randint(len(x['captions']), (1,))[0].item() for x in batched_inputs]
#         #     captions = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
#         #     loss_img_txt_level = self.image_text_matching(captions, global_feats=global_feats)
#         #     losses.update({'loss_img_txt_level': loss_img_txt_level})

#         # # region-concept level matching. All-zero background embedding is added to self.concept_emb 
#         # if self.concept_emb is not None:
#         #     self.region_concept_matching(images, proposals, gt_instances, region_feats, losses)     # inplace upate losses

#         ### vldet loss
#         # import ipdb
#         # ipdb.set_trace()
#         ann_type = 'box'
#         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         for inst, x in zip(gt_instances, batched_inputs):
#             inst._ann_type = x['ann_type']
#             inst._pos_category_ids = x['pos_category_ids']  # [] for 'box'
#         ann_types = [x['ann_type'] for x in batched_inputs]
#         assert len(set(ann_types)) == 1
#         ann_type = ann_types[0]

#         # localization branch: offline modules to get the region proposals
#         proposals = self.get_region_proposals(batched_inputs, random_pick=self.random_sample_region)

#         # recognition branch: get 2D feature maps using the backbone of recognition branch and extract region features
#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         losses = {}

#         # import ipdb
#         # ipdb.set_trace()
#         if self.img_txt_level:
#             ## box sup + ita loss
#             if 'caption' in ann_type:
#                 global_proposals = self.create_global_proposals(proposals)
#                 global_feats = self.get_region_features(images, features, global_proposals, gt_instances=None)
#                 # global_feats = self.get_CLIP_image_feats(batched_inputs) if ('caption' in ann_type) else None    # oringial clip features

#                 # encode text
#                 inds = [torch.randint(len(x['captions']), (1,))[0].item() \
#                     for x in batched_inputs]
#                 captions = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
#                 text = self.tokenizer(captions).to(self.device)
#                 caption_embs = self.lang_encoder.encode_text(text, only_eot=self.only_eot)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]
#                 # collect from other GPUs
#             else:
#                 global_feats = None
#                 caption_embs = None

#             loss_img_txt_level = self.roi_heads.sync_image_text_loss(global_feats, caption_embs, local_batch_size=len(batched_inputs), ann_type=ann_type)
#             losses.update({'loss_img_txt_level': loss_img_txt_level * self.image_loss_weight})

#             # # import ipdb
#             # # ipdb.set_trace()
#             # # gather from all GPUs. should be run on every gpu to avoid blocking
#             # global_feats_allGPU = self._gather_caption_features(global_feats)  # will include zero embeddings
#             # caption_embs_allGPU = self._gather_caption_features(caption_embs)

#             # # self.debug_count += 10
#             # # rank = torch.full((32, 1), self.debug_count + comm.get_rank(), dtype=torch.int32, device=self.device)  # for debug
#             # # rank_allGPU, _ = gather_tensors(rank)

#             # # # for debug
#             # # # if torch.isnan(row_loss).any() or torch.isnan(col_loss).any():
#             # # if self.debug:
#             # #     torch.save(
#             # #         {
#             # #             "global_feats": global_feats.detach().cpu() if global_feats is not None else None,
#             # #             "caption_embs": caption_embs.detach().cpu() if caption_embs is not None else None,
#             # #             "global_feats_allGPU": global_feats_allGPU.detach().cpu() if global_feats_allGPU is not None else None,
#             # #             "caption_embs_allGPU": caption_embs_allGPU.detach().cpu() if caption_embs_allGPU is not None else None,
#             # #             "rank_allGPU": rank_allGPU.detach().cpu(),
#             # #             "debug_count": self.debug_count,
#             # #         },
#             # #         "output/r50_pre_box_regionclip/debug_gather_%d.pth"%(comm.get_rank())
#             # #     )
#             # #     self.debug = False

#             # if 'caption' in ann_type:
#             #     # image-text level matching
#             #     # loss_img_txt_level = self.image_text_loss_box_captions(global_feats, caption_embs, 
#             #     #     global_feats_allGPU.detach().clone(), 
#             #     #     caption_embs_allGPU.detach().clone(), 
#             #     #     local_batch_size=len(batched_inputs)
#             #     # )
#             #     # gradients on the global batch
#             #     loss_img_txt_level = self.image_text_loss_box_captions(global_feats_allGPU, caption_embs_allGPU)
#             # else:
#             #     # fake value for ann_type == 'box'
#             #     loss_img_txt_level = images.tensor.new_zeros([])
#             # losses.update({'loss_img_txt_level': loss_img_txt_level})


#         ## for CLIPRes5ROIHeads
#         # # import ipdb
#         # # ipdb.set_trace()
#         # if self.weak_loss_type in ['ema_contrastive', 'ema_sinkhorn']:
#         #     self._momentum_update() # update the ema model 

#         assert self.use_clip_c4, 'current roi head not support' # use C4 + resnet weights from CLIP
#         if self.use_clip_attpool: # use att_pool from CLIP to match dimension
#             _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, ann_type=ann_type, ema_inputs=[self.teacher_backbone, self.teacher_roi_heads])
#         else: # use mean pool
#             _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, ann_type=ann_type, ema_inputs=[self.teacher_backbone, self.teacher_roi_heads])
#         losses.update(detector_losses)

#         # disable loss_cls if no binary foreground classification
#         if self.ignore_cls_loss:
#             # used in pretraining, disable roi head cls loss
#             losses['loss_cls'] *= 0

#         return losses

#     def get_region_features(self, images, features, proposals, gt_instances):
#         """ Input images and region proposals, return region features
#         """
#         # Given the proposals, crop region features from 2D image features
#         if self.use_clip_c4: # use C4 + resnet weights from CLIP
#             if self.use_clip_attpool: # use att_pool from CLIP to match dimension
#                 region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool, ann_type='feature_only')
#             else: # use mean pool
#                 region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, ann_type='feature_only')
#         else:  # regular detector setting
#             region_feats = self.roi_heads(images, features, proposals, gt_instances, ann_type='feature_only')
#         return region_feats

#     def get_CLIP_image_feats(self, batched_inputs):
#         """ Input images and region proposals, return region features
#         """
#         imagesTensors = []
#         for cur_input in batched_inputs:
#             cur_image = cur_input['image']  # CHW
#             # cur_image = cur_image.permute(1,2,0).cpu().numpy()   # CHW --> HWC, RGB, [0,255]
#             # cur_image = Image.fromarray(cur_image, mode="RGB")  # PIL obj
#             # cur_image = self.clip_transform(cur_image)  # 3 x 224 x 224
#             cur_image = tvt_F.resize(cur_image, 224, interpolation=InterpolationMode.BICUBIC, antialias=True)   # small edge --> 224
#             cur_image = self.center_crop(cur_image) # 3 x 224 x 224
#             if self.div_pixel:
#                 cur_image = ((cur_image.to(self.device) / 255.0) - self.pixel_mean) / self.pixel_std
#             else:
#                 cur_image = (cur_image.to(self.device) - self.pixel_mean) / self.pixel_std
#             imagesTensors.append(cur_image)

#         imagesTensors = torch.stack(imagesTensors, dim=0) # N x 3 x 224 x 224
#         image_feats = self.backbone(imagesTensors)['res4'] # res4, N x C x 14 x 14
#         image_feats = self.backbone.layer4(image_feats) # res5, N x C x 7 x 7
#         image_feats = self.backbone.attnpool(image_feats) # N x C, will get slight different results from CLIP due to slightly different running_var in bnx
#         return image_feats

#     def create_global_proposals(self, region_proposals=None):
#         """ create a single global box for an image, so as to extract global image features with RoIAlign on high-resolution images.
#         """
#         global_proposals = []
#         for iidx, cur_proposals in enumerate(region_proposals):
#             curH, curW = cur_proposals.image_size
#             curInst = Instances((curH, curW))
#             curInst.proposal_boxes = Boxes(cur_proposals.proposal_boxes.tensor.new_tensor([0, 0, curW, curH]).view(1, 4))
#             global_proposals.append(curInst)
#         return global_proposals

#     def get_region_proposals(self, batched_inputs, random_pick=False):
#         """ Given image, return object proposals
#         """
#         with torch.no_grad():  
#             if self.clip_crop_region_type == "RANDOM":  # from random proposals
#                 proposals = self.create_rand_boxes(batched_inputs)         
#             elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
#                 if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
#                     self.offline_backbone.eval() 
#                     self.offline_proposal_generator.eval()  
#                 images = self.offline_preprocess_image(batched_inputs)
#                 features = self.offline_backbone(images.tensor)
#                 if self.offline_proposal_generator is not None:
#                     proposals, _ = self.offline_proposal_generator(images, features, None)     
#             #visualize_proposals(batched_inputs, proposals, self.input_format, vis_pretrain=True)
        
#         # randomly select proposals
#         if random_pick:
#             rand_inds = [torch.randperm(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
#             proposals = [p[rand_inds[i]] for i, p in enumerate(proposals)]
#         return proposals

#     def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
#         Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
#         """
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         if self.div_pixel:
#             images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
#         else:
#             images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images

#     def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
#         Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
#         """
#         # import ipdb
#         # ipdb.set_trace()
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
#             (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
#             images = [x[[2,1,0],:,:] for x in images]
#         if self.offline_div_pixel:
#             images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
#         else:
#             images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
#         images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
#         return images

#     def inference(
#         self,
#         batched_inputs: List[Dict[str, torch.Tensor]],
#         detected_instances: Optional[List[Instances]] = None,
#         do_postprocess: bool = True,
#     ):
#         """
#         Run inference on the given inputs.

#         Args:
#             batched_inputs (list[dict]): same as in :meth:`forward`
#             detected_instances (None or list[Instances]): if not None, it
#                 contains an `Instances` object per image. The `Instances`
#                 object contains "pred_boxes" and "pred_classes" which are
#                 known boxes in the image.
#                 The inference will then skip the detection of bounding boxes,
#                 and only predict other per-ROI outputs.
#             do_postprocess (bool): whether to apply post-processing on the outputs.

#         Returns:
#             When do_postprocess=True, same as in :meth:`forward`.
#             Otherwise, a list[Instances] containing raw network outputs.
#         """
#         assert not self.training
        
#         # localization branch: offline modules to get the region proposals
#         if self.clip_crop_region_type == "GT":  # from ground-truth
#             proposals = []
#             for r_i, b_input in enumerate(batched_inputs): 
#                 this_gt = copy.deepcopy(b_input["instances"])  # Instance
#                 gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
#                 this_gt._fields = {'proposal_boxes': gt_boxes} #, 'objectness_logits': None}
#                 proposals.append(this_gt)                
#         elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
#             images = self.offline_preprocess_image(batched_inputs)
#             features = self.offline_backbone(images.tensor)
#             if detected_instances is None:
#                 if self.offline_proposal_generator is not None:
#                     proposals, _ = self.offline_proposal_generator(images, features, None)     

#         # import ipdb
#         # ipdb.set_trace()
#         # ## debug teacher model
#         # images = self.preprocess_image(batched_inputs)
#         # features = self.teacher_backbone(images.tensor)
#         # # Given the proposals, crop region features from 2D image features and classify the regions
#         # results, _ = self.teacher_roi_heads(images, features, proposals, None, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)

#         ## original eval
#         # recognition branch: get 2D feature maps using the backbone of recognition branch
#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         # Given the proposals, crop region features from 2D image features and classify the regions
#         if self.use_clip_c4: # use C4 + resnet weights from CLIP
#             if self.use_clip_attpool: # use att_pool from CLIP to match dimension
#                 results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
#             else: # use mean pool
#                 results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
#         else:  # regular detector setting
#             if self.use_clip_attpool: # use att_pool from CLIP to match dimension
#                 results, _  = self.roi_heads(images, features, proposals, None, attnpool=self.backbone.bottom_up.attnpool)
#             else:
#                 results, _  = self.roi_heads(images, features, proposals, None)
        
#         #visualize_proposals(batched_inputs, proposals, self.input_format)
#         if do_postprocess:
#             assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
#             return CLIPFastRCNN._postprocess(results, batched_inputs)
#         else:
#             return results


