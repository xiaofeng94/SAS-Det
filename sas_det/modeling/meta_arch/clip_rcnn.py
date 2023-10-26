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
                    # self.pl_threshold = min(self.pl_threshold + 0.05, 1)
                    self.pl_threshold = torch.clamp(self.pl_threshold + self.adaptive_thres_delta, min=0., max=1.)
                
                # reset
                self.PLs_inst_num = 0
                self.PLs_count = 0
 
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

            with torch.no_grad():  
                gt_instances = self.add_ovd_PLs_from_teacher(images, proposals, gt_instances=gt_instances, ann_type=ann_type, base_cat_ids=self.base_cat_ids)

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

        if self.eval_pseudo_labels:
            ## eval pseudo labels and output avg # PLs
            results = self.add_ovd_PLs_from_teacher(images, proposals, base_cat_ids=self.base_cat_ids)
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
