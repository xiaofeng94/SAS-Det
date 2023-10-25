import argparse
import json

from lvis import LVIS
from lvis import LVISEval, LVISResults

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate PLs quality offline')
    parser.add_argument('gt_json', type=str, help='gt coco json file')
    parser.add_argument('pl_json', type=str, help='PL coco json file')
    args = parser.parse_args()
    # print(args)

    #############################################
    gt_LVISJson_file = args.gt_json
    pred_LvisJson_file = args.pl_json   
    
    covert_to_result = True # True if .json in coco data format (not coco result format)

    #############################################

    # load image list in gt_json
    lvis_gt = LVIS(gt_LVISJson_file)
    gt_img_ids = set(lvis_gt.get_img_ids())

    if covert_to_result:
        PLData = json.load(open(pred_LvisJson_file, 'r'))
        PL_list = list()
        imageId_list = list()
        for anno in PLData['annotations']:
            cur_image_id = anno['image_id']
            ## eval only on PLs
            if ("thing_isNovel" in anno.keys()) and anno['thing_isNovel'] and (cur_image_id in gt_img_ids):
                data = {'image_id': cur_image_id,
                        'category_id': anno['category_id'],
                        'bbox': anno['bbox'],
                        'score': anno['confidence']}
                PL_list.append(data)
                imageId_list.append(cur_image_id)
            # ## eval on all data (GT + PLs)
            # if cur_image_id in gt_img_ids:
            #     data = {'image_id': cur_image_id,
            #             'category_id': anno['category_id'],
            #             'bbox': anno['bbox'],
            #             'score': anno['confidence']}
            #     PL_list.append(data)
            #     imageId_list.append(cur_image_id)

        # import ipdb
        # ipdb.set_trace()
        print( 'Total PL boxes num: %d, avg num: %.2f\n' % (len(PL_list), len(PL_list)/len(set(imageId_list))) )
    else:
        PL_list = json.load(open(pred_LvisJson_file, 'r'))

    # import ipdb
    # ipdb.set_trace()
    # do evaluation
    lvis_results = LVISResults(lvis_gt, PL_list, max_dets=300)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type="bbox")
    lvis_eval.run()
    lvis_eval.print_results()
