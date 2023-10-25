# Improving Pseudo Labels for Open-Vocabulary Object Detection

Official implementation of online self-training and a split-and-fusion (SAF) head for Open-Vocabulary Object Detection (OVD), SAS-Det for short.

[arXiv](https://arxiv.org/abs/2308.06412)


## Installation

Download and install [Detectron2](https://github.com/facebookresearch/detectron2). We recomend to put Detectron2 root folder into the root folder of this project.


## Datasets

- Please follow [RegionCLIP's instructions](https://github.com/microsoft/RegionCLIP/blob/main/datasets/README.md) to prepare COCO and LVIS datasets.

- Download and put [metadata](https://drive.google.com/drive/u/1/folders/1R72q0Wg26-PQGqbaK3P3pT2vmGm9uzKU) for datasets in the folder `datasets` (i.e., `$DETECTRON2_DATASETS` used in the last step), which will be used in our evaluation and training.


## Download pretrained weights
- Download various RegionCLIP's [pretrained weights](https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii). Check [here](https://github.com/microsoft/RegionCLIP/blob/main/docs/MODEL_ZOO.md#model-downloading) for more details.
Create a new folder `pretrained_ckpt` to put those weights. In this repository, `regionclip`, `concept_emb` and `rpn` will be used.

- Download our [pretrained weights](https://drive.google.com/drive/u/1/folders/1TAr7nZSvpB6nCZCC6nXBw6xgmMmlL0X9) and put them in corresponding folders in `pretrained_ckpt`.


## Evaluation with released weights

### Results on COCO-OVD
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">Novel AP</th>
<th valign="bottom">Base AP</th>
<th valign="bottom">Overall AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr><td align="left"><a href="./myRegionCLIP/configs/COCO-InstanceSegmentation/vldet/CLIP_fast_rcnn_R_50_C4_ovd_PLs.yaml">w/o SAF head</a></td>
<td align="center">31.4</td>
<td align="center">55.7</td>
<td align="center">49.4</td>
<td align="center">/net/acadia4a/data/szhao/code/RegionCLIP/output/ovd_coco_defRegClip_ft_PLs_periodic_boxConf/model_final.pth</td>
</tr>
<!-- ROW: with out LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_coco_R50_C4_ensemble_PLs.yaml">with SAF head</a></td>
<td align="center">37.4</td>
<td align="center">58.5</td>
<td align="center">53.0</td>
<td align="center">/net/acadia4a/data/szhao/code/RegionCLIP/output/ovd_coco_cocoRPN_ensemble_PLs_attn/model_final.pth</td>
</tr>
</tbody></table>

<details>
<summary>
Evaluation without the SAF Head,
</summary>
  
```bash
python3 ./train_net_ensemble.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ./myRegionCLIP/configs/COCO-InstanceSegmentation/vldet/CLIP_fast_rcnn_R_50_C4_ovd_PLs.yaml \
    MODEL.WEIGHTS <path_to_your_weights> \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./myRegionCLIP/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
    MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
    OUTPUT_DIR output/eval
```
</details>

<details>
<summary>
Evaluation with the SAF Head,
</summary>
  
```bash
python3 ./train_net_ensemble.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ./sas_det/configs/ovd_coco_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS <path_to_your_weights> \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./myRegionCLIP/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_coco_48_base_17_cls_emb.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
    MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/coco_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.33 MODEL.ENSEMBLE.BETA 0.67 \
    OUTPUT_DIR output/eval
```
</details>


### Results on LVIS-OVD
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">APr</th>
<th valign="bottom">APc</th>
<th valign="bottom">APf</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: with LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml">RN50-C4 as backbone</a></td>
<td align="center">20.1</td>
<td align="center">27.1</td>
<td align="center">32.9</td>
<td align="center">28.1</td>
<td align="center">/net/acadia4a/data/szhao/code/SAS-Det/output/ovd_lvis_ensemble_PLs_attn_re2/model_final.pth</td>
</tr>
<!-- ROW: with out LSJ -->
 <tr><td align="left"><a href="./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml">RN50x4-C4 as backbone</a></td>
<td align="center">29.0</td>
<td align="center">32.3</td>
<td align="center">36.8</td>
<td align="center">33.5</td>
<td align="center">/net/acadia4a/data/szhao/code/SAS-Det/output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/model_final.pth</td>
</tr>
</tbody></table>

<details>
<summary>
Evaluation with RN50-C4 as the backbone,
</summary>
  
```bash
python3 ./train_net_ensemble.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ./sas_det/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS <path_to_your_weights> \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./myRegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_lvis_866_base_337_cls_emb.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
    MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/lvis_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.33 MODEL.ENSEMBLE.BETA 0.67 \
    OUTPUT_DIR output/eval
```
</details>

<details>
<summary>
Evaluation with RN50x4-C4 as the backbone,
</summary>
  
```bash
python3 ./train_net_ensemble.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ./detpro/configs/ovd_lvis_R50_C4_ensemble_PLs.yaml \
    MODEL.WEIGHTS <path_to_your_weights> \
    MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866_lsj.pth \
    MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_866_base_cls_emb_rn50x4.pth \
    MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_lvis_866_base_337_cls_emb_rn50x4.pth \
    MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
    MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED True \
    MODEL.CLIP.TEXT_EMB_DIM 640 \
    MODEL.RESNETS.DEPTH 200 \
    MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
    MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION 18 \
    MODEL.ENSEMBLE.TEST_CATEGORY_INFO "datasets/lvis_ovd_continue_cat_ids.json" \
    MODEL.ENSEMBLE.ALPHA 0.33 MODEL.ENSEMBLE.BETA 0.67 \
    OUTPUT_DIR output/eval
```
</details>



## Acknowledgement

This repository was built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [RegionCLIP](https://github.com/microsoft/RegionCLIP), and [VLDet](https://github.com/clin1223/VLDet). We thank the effort from our community.