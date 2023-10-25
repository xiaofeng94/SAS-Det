# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets.builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
# from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
# from .cityscapes_panoptic import register_all_cityscapes_panoptic
from detectron2.data.datasets.coco import load_sem_seg, register_coco_instances
# from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
# from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances
# from .pascal_voc import register_pascal_voc

from .lvis import get_lvis_instances_meta, register_lvis_instances_w_PLs, register_lvis_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
# _PREDEFINED_SPLITS_COCO["coco"] = {
#     "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
#     "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
#     "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
#     "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
#     "coco_2014_valminusminival": (
#         "coco/val2014",
#         "coco/annotations/instances_valminusminival2014.json",
#     ),
#     "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
#     "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
#     "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
#     "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
#     "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
# }
_PREDEFINED_SPLITS_COCO["coco_ovd"] = {
    "coco_2017_ovd_all_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_all.json"),
    "coco_2017_ovd_b_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_b.json"),
    "coco_2017_ovd_b_train_65cats": ("coco/train2017", "coco/annotations/ovd_ins_train2017_b_65cats.json"),
    "coco_2017_ovd_b_train_65cats_all_images": ("coco/train2017", "coco/annotations/ovd_ins_train2017_b_65cats_all_images.json"),
    "coco_2017_ovd_t_train": ("coco/train2017", "coco/annotations/ovd_ins_train2017_t.json"),
    #
    "coco_2017_ovd_all_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_all.json"),
    "coco_2017_ovd_b_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_b.json"),
    "coco_2017_ovd_t_test": ("coco/val2017", "coco/annotations/ovd_ins_val2017_t.json"),
    #
    "coco_2017_ovd_retain_val": ("coco/val2017", "coco/annotations/ovd_ins_val2017_retain_15.json"),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        if dataset_name == 'coco_ovd':  # for zero-shot split
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    {}, # empty metadata, it will be overwritten in load_coco_json() function
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )
        else: # default splits
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    _get_builtin_metadata(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    # # openset setting
    # "lvis_v1": {
    #     "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
    #     "lvis_v1_train_p0": ("coco/", "lvis/lvis_v1_train_p0.json"),
    #     "lvis_v1_train_p1": ("coco/", "lvis/lvis_v1_train_p1.json"),
    #     "lvis_v1_train_p2": ("coco/", "lvis/lvis_v1_train_p2.json"),
    #     "lvis_v1_train_p3": ("coco/", "lvis/lvis_v1_train_p3.json"),
    #     #
    #     "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
    #     "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
    #     "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    # },
    # custom image setting
    "lvis_v1_custom_img": {
        "lvis_v1_train_custom_img": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_custom_img": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    # regular fully supervised setting
    "lvis_v1_fullysup": {
        "lvis_v1_train_fullysup": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_fullysup": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
        #
        "lvis_v1_train_base_1203cats": ("coco/", "lvis/lvis_v1_train_baseOnly.json"),
        "lvis_v1_val_1@10": ("coco/", "lvis/lvis_v1_val_1@10.json"),
    },
    # PLs for ensemble by zsy
    "lvis_v1_PLs": {
        "lvis_v1_train_base_PLs_r50x4": ("coco/", "lvis/regionclip_PLs/inst_train_defRegCLIPr50x4_PLs_93.json"),
        "lvis_v1_train_SASDet_r50x4_PLs": ("coco/", "lvis/regionclip_PLs/lvis_v1_train_SASDet_r50x4_PLs_t62.json"),
        "lvis_v1_o365_SASDet_r50x4_PLs": ("Objects365/train", "Objects365/regionclip_PLs/zsy_objv1_train_SASDet_r50x4_PLs_t83.json"),
    }
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        if dataset_name == "lvis_v1_PLs":
            for key, (image_root, json_file) in splits_per_dataset.items():
                register_lvis_instances_w_PLs(
                    key,
                    get_lvis_instances_meta(dataset_name),  # TODO: meta for PLs, category order is rearranged
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )
        else:
            for key, (image_root, json_file) in splits_per_dataset.items():
                if dataset_name == "lvis_v1":
                    args = {'filter_open_cls': True, 'run_custom_img': False}
                elif dataset_name == 'lvis_v1_custom_img':
                    args = {'filter_open_cls': False, 'run_custom_img': True}
                elif dataset_name == 'lvis_v1_fullysup':
                    args = {'filter_open_cls': False, 'run_custom_img': False}
                register_lvis_instances(
                    key,
                    get_lvis_instances_meta(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                    args,
                )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
register_all_lvis(_root)

# # True for open source;
# # Internally at fb, we register them elsewhere
# if __name__.endswith(".builtin"):
#     # Assume pre-defined datasets live in `./datasets`.
#     _root = os.getenv("DETECTRON2_DATASETS", "datasets")
#     register_all_coco(_root)
#     register_all_lvis(_root)