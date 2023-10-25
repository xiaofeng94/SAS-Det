import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate PLs quality offline')
    parser.add_argument('gt_json', type=str, help='gt coco json file')
    parser.add_argument('pl_json', type=str, help='PL coco json file')
    parser.add_argument('-r', '--raw', action='store_true')

    args = parser.parse_args()
    # print(args)

    #############################################
    gt_COCOJson_file = args.gt_json
    pred_COCOJson_file = args.pl_json   
    #############################################

    # load image list in gt_json
    GtData = json.load(open(gt_COCOJson_file, 'r'))
    gt_img_ids = [x['id'] for x in GtData['images']]
    gt_img_ids = set(gt_img_ids)

    PLData = json.load(open(pred_COCOJson_file, 'r'))
    
    # import ipdb
    # ipdb.set_trace()
    if args.raw:
        PL_list = PLData
        imageId_list = gt_img_ids
    else:
        PL_list = list()
        imageId_list = list()
        for anno in PLData['annotations']:
            cur_image_id = anno['image_id']

            score = anno.get('confidence', None)
            if score is None:
                # take all PLs
                data = {'image_id': cur_image_id,
                        'category_id': anno['category_id'],
                        'bbox': anno['bbox'],
                        'score': anno['confidence']}
                PL_list.append(data)
                imageId_list.append(cur_image_id)

            # if args.raw:
            #     # take all annos from PLs
            #     data = {'image_id': cur_image_id,
            #             'category_id': anno['category_id'],
            #             'bbox': anno['bbox'],
            #             'score': anno['confidence']}
            #     PL_list.append(data)
            #     imageId_list.append(cur_image_id)
            # else:
            #     if ("thing_isNovel" in anno.keys()) and anno['thing_isNovel'] and (cur_image_id in gt_img_ids):
            #         data = {'image_id': cur_image_id,
            #                 'category_id': anno['category_id'],
            #                 'bbox': anno['bbox'],
            #                 'score': anno['confidence']}
            #         PL_list.append(data)
            #         imageId_list.append(cur_image_id)

    # import ipdb
    # ipdb.set_trace()
    print( 'Total PL boxes num: %d, avg num: %.2f\n' % (len(PL_list), len(PL_list)/len(set(imageId_list))) )

    # import ipdb
    # ipdb.set_trace()
    curSaveJson = './.temp.json'
    with open(curSaveJson, 'w') as outfile:
        json.dump(PL_list, outfile)

    cocoGt = COCO(gt_COCOJson_file)
    cocoDt = cocoGt.loadRes(curSaveJson)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
