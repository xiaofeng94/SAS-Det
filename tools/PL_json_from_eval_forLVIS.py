import os
import json
from tqdm import tqdm 

import numpy as np
import cv2
import torch

import pycocotools.mask as mask_util


def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py

    # import ipdb
    # ipdb.set_trace()
    # clossing gap
    kernel = np.ones((10, 10), dtype=np.uint8)
    # dilateMask = cv2.dilate(maskedArr*255, kernel, 1)
    cur_mask = cv2.morphologyEx(maskedArr, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())

    if len(segmentation) == 0:
        return []

    # RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    # RLE = mask_util.merge(RLEs)
    # # RLE = mask.encode(np.asfortranarray(maskedArr))
    # area = mask_util.area(RLE)
    # [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0] #, [x, y, w, h], area


if __name__ == "__main__":
    #############################################################
    ## val data
    # # eval_json = "./output/ovd_lvis_r50x4_ensemble_PLs_attn/eval/inference/lvis_val_instances_results.json"
    # eval_json = "./output/ovd_lvis_default_regclip_ft_PLs_boxConf_re/eval3/inference/lvis_instances_results.json"
    # base_gt_json = "./datasets/lvis/lvis_v1_val.json"
    # ## train data
    # eval_json = "./output/ovd_lvis_r50x4_ensemble_PLs_attn/eval/inference/lvis_train_instances_results.json"
    # base_gt_json = "./datasets/lvis/lvis_v1_train.json"

    eval_json_list = [
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/lvis_train_v2_p0/inference/lvis_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/lvis_train_v2_p1/inference/lvis_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/lvis_train_v2_p2/inference/lvis_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/lvis_train_v2_p3/inference/lvis_instances_results.json",
    ]
    base_gt_json_list = [
        "./datasets/lvis/lvis_v1_train_p0.json",
        "./datasets/lvis/lvis_v1_train_p1.json",
        "./datasets/lvis/lvis_v1_train_p2.json",
        "./datasets/lvis/lvis_v1_train_p3.json",
    ]

    ovd_cat_info = "./datasets/lvis_ovd_continue_cat_ids.json"      # store continuous ids not coco id

    # for un-normalized scores
    normalize_score = True
    score_thres = 0.62 

    save_json = True
    # save_json_dir = "./output/ovd_lvis_r50x4_ensemble_PLs_attn/eval/inference/"
    # save_json_file = os.path.join(save_json_dir, "temp_PLs.json")
    save_json_dir = "./datasets/lvis/regionclip_PLs"
    save_json_file = os.path.join(save_json_dir, "lvis_v1_train_SASDet_r50x4_PLs_t62.json")

    #############################################################

    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)

    # evalResults = json.load(open(eval_json, "r"))
    evalResults = []
    for eval_json in eval_json_list:
        cur_predictions = json.load(open(eval_json, "r"))
        evalResults += cur_predictions

    # baseGtData = json.load(open(base_gt_json, "r"))
    baseGtData = None
    for base_gt_json in base_gt_json_list:
        if baseGtData is None:
            baseGtData = json.load(open(base_gt_json, "r"))
        else:
            # concat current ones
            cur_gt_data = json.load(open(base_gt_json, "r"))
            baseGtData['images'] += cur_gt_data['images']
            baseGtData['annotations'] += cur_gt_data['annotations']

    ovdCatInfo = json.load(open(ovd_cat_info, "r"))
    ovdCatInfo = ovdCatInfo['categories']
    novelCatLvisIds = set([x['id'] for x in ovdCatInfo if x['frequency'] == 'r'])
    catId2freq_map = {x['id']:x['frequency'] for x in ovdCatInfo}

    # Eval result sample
    # {'image_id': 632, 'category_id': 84, 'bbox': [431.2176818847656, 145.65663146972656, 31.380096435546875, 37.46173095703125], 'score': 0.64728844165802}

    # # Template for PLs
    # {
    #     'segmentation': [[]], 
    #     # 'iscrowd': 0, # lvis does not have this field
    #     'image_id': 491851, 
    #     'bbox': [550.6527099609375, 4.182845592498779, 13.72369384765625, 40.56206178665161],     # xyhw
    #     'area': 0, 
    #     'category_id': 49, 
    #     'id': -395478, 
    #     'confidence': 0.812894880771637, 
    #     'thing_isBase': False, 
    #     'thing_isNovel': True
    # }
    # # LVIS annotations
    # >>> data['annotations'][0].keys()
    # dict_keys(['area', 'id', 'segmentation', 'image_id', 'bbox', 'category_id'])

    # import ipdb
    # ipdb.set_trace()
    # for statistics
    num_PLs = 0
    # image_id_w_PLs = []

    cur_PL_box_id = -1
    new_annotations = []

    for eval_res in tqdm(evalResults):
        cur_cat_id = eval_res['category_id']    # lvis id

        if catId2freq_map[cur_cat_id] == 'r':
            new_anno = {}
            cur_image_id = eval_res['image_id']
            # image_id_w_PLs.append(cur_image_id)

            confid_score = eval_res['score']
            if normalize_score:
                confid_score = torch.sigmoid(torch.tensor(confid_score)).item()

            if confid_score > score_thres:
                new_anno['image_id'] = cur_image_id

                curBox = eval_res['bbox']
                new_anno['bbox'] = curBox   # xywh
                # new_anno['area'] = curBox[2] * curBox[3]

                # segment = mask_util.decode(eval_res['segmentation'])  # covnert rel format into binary mask
                # seg_area = (segment > 0).sum()
                # if seg_area <= 30:
                #     # create fake segment
                #     intX,intY,intW,intH = [int(x) for x in curBox]
                #     segment[intY:intY+intH, intX:intX+intW] = 1
                #     print('seg_area is too small, use fake segement')
                # segment = polygonFromMask(segment*255)
                
                # use fake segment
                seg_area = curBox[2] * curBox[3]
                fX, fY, fW, fH = [int(x) for x in curBox]
                segment = [fX, fY, fX+fW, fY, fX+fW, fY+fH, fX, fY+fH]

                new_anno['area'] = seg_area     # not precise
                new_anno['segmentation'] = [segment]  # [[...]]

                new_anno['category_id'] = cur_cat_id
                new_anno['id'] = cur_PL_box_id
                new_anno['confidence'] = confid_score
                new_anno['thing_isBase'] = False
                new_anno['thing_isNovel'] = True

                new_annotations.append(new_anno)
                cur_PL_box_id = cur_PL_box_id - 1
                num_PLs += 1

    # # LVIS annotations
    # >>> data['annotations'][0].keys()
    # dict_keys(['area', 'id', 'segmentation', 'image_id', 'bbox', 'category_id'])
    # {'segmentation': [[279.11, 370.25, 281.96, 214.56, 283.86, 209.81, 481.33, 215.51, 485.13, 370.25]], 'area': 31935.485049999996, 'image_id': 464476, 'bbox': [279.11, 209.81, 206.02, 160.44], 'category_id': 72, 'id': 29131}
    # add base gt annos
    # import ipdb
    # ipdb.set_trace()
    for anno in baseGtData['annotations']:
        if catId2freq_map[anno['category_id']] != 'r':
            anno['confidence'] = 1.0
            anno['thing_isBase'] = True
            anno['thing_isNovel'] = False
            new_annotations.append(anno)

    new_data = {
        'info': baseGtData.get('info', []), 
        'licenses': baseGtData.get('licenses', []), 
        'images': baseGtData['images'], 
        'annotations': new_annotations, 
        'categories': ovdCatInfo, 
    }

    # some statistics
    print('- score_thres: %.4f'%score_thres)
    print("# images: %d" % len(baseGtData['images']))
    print("# PLs: %d" % num_PLs)
    print('avg # PLs per image (on all images): %.04f' % (num_PLs/len(baseGtData['images'])))
    # print('avg # inst per image: %.04f' % (num_PLs/len(set(image_id_w_PLs))))

    # import ipdb
    # ipdb.set_trace()
    if save_json:
        print('saving to %s'%save_json_file)
        json.dump(new_data, open(save_json_file, 'w'))
 

