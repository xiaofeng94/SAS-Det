import os
import json
import itertools
from tqdm import tqdm 

import torch


if __name__ == "__main__":
    #############################################################
    # ## val data
    # eval_json_list = [
    #     "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p0/inference/coco_instances_results.json",
    #     "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p1/inference/coco_instances_results.json",
    #     "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p2/inference/coco_instances_results.json",
    #     "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p3/inference/coco_instances_results.json",
    # ]
    # # eval_pth = "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p0/inference/instances_predictions.pth"
    # base_gt_json_list = [
    #     "./datasets/Objects365/zsy_objv1_val_inLvisCat_allLvisCat_p0.json",
    #     "./datasets/Objects365/zsy_objv1_val_inLvisCat_allLvisCat_p1.json",
    #     "./datasets/Objects365/zsy_objv1_val_inLvisCat_allLvisCat_p2.json",
    #     "./datasets/Objects365/zsy_objv1_val_inLvisCat_allLvisCat_p3.json",
    # ]

    ## train data
    eval_json = "./output/ovd_coco_defRegClip_ft_PLs_per4k_clsBoxConf/inference/coco_train_instances_results.json"
    base_gt_json = "./datasets/coco/annotations/ovd_ins_train2017_b.json"
    eval_json_list = [
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_train_p0/inference/coco_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_train_p1/inference/coco_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_train_p2/inference/coco_instances_results.json",
        "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_train_p3/inference/coco_instances_results.json",
    ]
    # eval_pth = "./output/ovd_lvis_r50x4_ensemble_offPLs_attn_re/PLs/PLs_on_val_p0/inference/instances_predictions.pth"
    base_gt_json_list = [
        "./datasets/Objects365/zsy_objv1_train_inLvisCat_allLvisCat_p0.json",
        "./datasets/Objects365/zsy_objv1_train_inLvisCat_allLvisCat_p1.json",
        "./datasets/Objects365/zsy_objv1_train_inLvisCat_allLvisCat_p2.json",
        "./datasets/Objects365/zsy_objv1_train_inLvisCat_allLvisCat_p3.json",
    ]

    o365_cat_info = "./datasets/Objects365/o365_catId_to_lvis_info.json"
    lvis_cat_info = "./datasets/lvis_ovd_continue_cat_ids.json"      # store continuous ids not coco id

    # for un-normalized scores
    normalize_score = True
    novel_score_thres = 0.3
    base_score_thres = 0.83

    save_json = True
    save_json_dir = "./datasets/Objects365/regionclip_PLs/"
    save_json_file = os.path.join(save_json_dir, "zsy_objv1_train_SASDet_r50x4_PLs_t83.json")

    #############################################################

    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)

    # LVIS categories info
    lvis_cat_infoList = json.load(open(lvis_cat_info, "r"))
    lvis_cat_infoList = lvis_cat_infoList['categories']     # assume already sorted
    # cat id to frequency, 'f', 'c', or 'r'
    lvisCatId_to_freq_dict = {}
    # lvis_id_to_contiguous_id = {}
    for cidx, catInfo in enumerate(lvis_cat_infoList):
        cur_cat_id = catInfo['id']
        lvisCatId_to_freq_dict[cur_cat_id] = catInfo['frequency']
        # lvis_id_to_contiguous_id[cur_cat_id] = cidx

    # import ipdb
    # ipdb.set_trace()  
    # combine all predictions
    evalResults = []
    for eval_json in eval_json_list:
        cur_predictions = json.load(open(eval_json, "r"))
        evalResults += cur_predictions

    # import ipdb
    # ipdb.set_trace()
    # predictions = torch.load(eval_pth)  # w/ continous cat id
    # evalResults = list(itertools.chain(*[x["instances"] for x in predictions]))
    # # continous cat id --> lvis cat id
    # reverse_id_mapping = {v: k for k, v in lvis_id_to_contiguous_id.items()}
    # for result in evalResults:
    #     category_id = result["category_id"]
    #     result["category_id"] = reverse_id_mapping[category_id]

    # import ipdb
    # ipdb.set_trace()    
    # original annotations from objects365
    # combine all splits
    baseGtData = None
    for base_gt_json in base_gt_json_list:
        if baseGtData is None:
            baseGtData = json.load(open(base_gt_json, "r"))
        else:
            # concat current ones
            cur_gt_data = json.load(open(base_gt_json, "r"))
            baseGtData['images'] += cur_gt_data['images']
            baseGtData['annotations'] += cur_gt_data['annotations']

    # map from Objects365 category id to LVIS category id. NOTE: not all classes objects365 are in LVIS
    o365_catId_to_lvisCatInfo_dict = {}
    # lvis categories that exists in objects365
    lvisCats_in_o365 = []
    
    ovdCatInfo = json.load(open(o365_cat_info, "r"))
    for _, cur_cat_info in ovdCatInfo.items():
        o365_id = cur_cat_info['o365_id']
        lvisCatInfo = cur_cat_info['lvis_cat']

        if (len(lvisCatInfo) > 0) and ('id' in lvisCatInfo):
            o365_catId_to_lvisCatInfo_dict[o365_id] = lvisCatInfo['id']
            lvisCats_in_o365.append(lvisCatInfo['id'])

    # multiple o365 cats may be mappped to the same lvis cat
    lvisCats_in_o365 = sorted(list(set(lvisCats_in_o365)))

    # Eval result sample
    # {'image_id': 632, 'category_id': 84, 'bbox': [431.2176818847656, 145.65663146972656, 31.380096435546875, 37.46173095703125], 'score': 0.64728844165802}

    # # Template for PLs
    # {
    #     'segmentation': [[]], 
    #     'iscrowd': 0, 
    #     'image_id': 491851, 
    #     'bbox': [550.6527099609375, 4.182845592498779, 13.72369384765625, 40.56206178665161],     # xyhw
    #     'area': 0, 
    #     'category_id': 49, 
    #     'id': -395478, 
    #     'confidence': 0.812894880771637, 
    #     'thing_isBase': False, 
    #     'thing_isNovel': True
    # }

    # for statistics
    num_base_PLs = 0
    num_novel_PLs = 0
    image_id_w_PLs = []

    unified_anno_id = 0
    new_annotations = []

    # import ipdb
    # ipdb.set_trace()
    for eval_res in tqdm(evalResults):
        cur_cat_id = eval_res['category_id']    # LVIS id
        if cur_cat_id in lvisCats_in_o365:
            # we have GT annotations for those cats, PLs are no need
            continue

        new_anno = {'iscrowd': 0}
        cur_image_id = eval_res['image_id']     # image id in objects365
        image_id_w_PLs.append(cur_image_id)

        confid_score = eval_res['score']
        if normalize_score:
            confid_score = torch.sigmoid(torch.tensor(confid_score)).item()

        # we set different threshold for LVIS novel and base cats
        use_this_PL = False
        if lvisCatId_to_freq_dict[cur_cat_id] == 'r':
            # PLs for novel
            if confid_score > novel_score_thres:
                use_this_PL = True
        else:
            # PLs for base
            if confid_score > base_score_thres:
                use_this_PL = True

        if use_this_PL:
            new_anno['image_id'] = cur_image_id

            curBox = eval_res['bbox']
            new_anno['bbox'] = curBox

            # use fake segment
            seg_area = curBox[2] * curBox[3]
            fX, fY, fW, fH = [int(x) for x in curBox]
            segment = [fX, fY, fX+fW, fY, fX+fW, fY+fH, fX, fY+fH]

            new_anno['area'] = seg_area     # not precise
            new_anno['segmentation'] = [segment]  # [[...]]

            new_anno['category_id'] = cur_cat_id
            new_anno['id'] = unified_anno_id
            new_anno['confidence'] = confid_score

            if lvisCatId_to_freq_dict[cur_cat_id] == 'r':
                # PLs for novel
                new_anno['thing_isBase'] = False
                new_anno['thing_isNovel'] = True
                num_novel_PLs += 1
            else:
                # PLs for base
                new_anno['thing_isBase'] = True
                new_anno['thing_isNovel'] = False
                num_base_PLs += 1

            new_annotations.append(new_anno)
            unified_anno_id = unified_anno_id + 1



    # import ipdb
    # ipdb.set_trace()
    # Objects365 annotation example: 
    # {'id': 27943730, 'iscrowd': 1, 'isfake': 0, 'area': 14850.210039605172, 'isreflected': 0, 'bbox': [2.4785156096, 241.2398681353, 107.59851074560001, 138.0150146754], 'image_id': 452184, 'category_id': 1}
    # add objects gt annos
    num_o365_GT = 0
    for anno in baseGtData['annotations']:
        cur_cat_id = anno['category_id']    # LVIS id
        
        # o365_cat_id = anno['category_id']    # objects365 id
        # if o365_cat_id in o365_catId_to_lvisCatInfo_dict:
            # # only take GT annotations whose cats are in LVIS
            # cur_cat_id = o365_catId_to_lvisCatInfo_dict[o365_cat_id]     # LVIS id

        if lvisCatId_to_freq_dict[cur_cat_id] == 'r':
            # ignore those to avoid data leakage
            # # GT for novel
            # anno['thing_isBase'] = False
            # anno['thing_isNovel'] = True
            continue

        anno['category_id'] = cur_cat_id    # objects365 id --> LVIS id
        anno['id'] = unified_anno_id
        anno['confidence'] = 1.0

        fX, fY, fW, fH = [int(x) for x in anno['bbox']]
        segment = [fX, fY, fX+fW, fY, fX+fW, fY+fH, fX, fY+fH]
        anno['segmentation'] = [segment]  # [[...]]

        # GT for base
        anno['thing_isBase'] = True
        anno['thing_isNovel'] = False
        new_annotations.append(anno)
        unified_anno_id = unified_anno_id + 1
        num_o365_GT += 1

    # filter out images w/ no annotations
    img_ids_w_annos = set()
    cat_ids_w_annos = set()
    for anno in new_annotations:
        img_ids_w_annos.add(anno['image_id'])
        cat_ids_w_annos.add(anno['category_id'])

    new_imageInfo = []
    for imgInfo in baseGtData['images']:
        if imgInfo['id'] in img_ids_w_annos:
            new_imageInfo.append(imgInfo)

    new_data = {
        'info': baseGtData.get('info', []), 
        'licenses': baseGtData.get('licenses', []), 
        'images': new_imageInfo, 
        'annotations': new_annotations, 
        'categories': lvis_cat_infoList, 
    }

    # some statistics
    print('- novel_score_thres: %.4f, base_score_thres: %.4f ' % (novel_score_thres, base_score_thres))
    print("# oringinal images: %d, images w/ annos: %d" % (len(baseGtData['images']), len(new_imageInfo)))
    print("# used o365 GT: %d out of %d (total in o365)" % (num_o365_GT, len(baseGtData['annotations'])))

    #
    image_id_w_PLs = list(set(image_id_w_PLs))
    print("# base PLs: %d, # novel PLs: %d" % (num_base_PLs, num_novel_PLs) )
    print('avg # base PLs per image (w/ PLs): %.04f (%d/%d)' % (num_base_PLs/len(image_id_w_PLs), num_base_PLs, len(image_id_w_PLs)))
    print('avg # novel PLs per image (w/ PLs): %.04f (%d/%d)' % (num_novel_PLs/len(image_id_w_PLs), num_novel_PLs, len(image_id_w_PLs)))

    # import ipdb
    # ipdb.set_trace()
    if save_json:
        print('saving to %s'%save_json_file)
        json.dump(new_data, open(save_json_file, 'w'))
 

