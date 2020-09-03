import os
import numpy as np

from tqdm import tqdm

import sys
sys.path.append('./mmdetection/')
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector
from constant import *


def infer(config, checkpoint, img_file_dir, output_dir, json_name='bbox_score.json', show_score_thr=0.3):

    model = init_detector(config, checkpoint, device=device)
    img_dir = img_file_dir
    file_name_list = os.listdir(img_dir)
    #img_dir2 = img_dir.replace('_p', '')
    img_dir2 = img_dir.split("_")[0]
    results = {}
    ik = 0
    for i in tqdm(range(len(file_name_list))):
        file_name = file_name_list[i]
        if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
            continue
        result_p = inference_detector(model, os.path.join(img_dir, file_name))
        result_c = inference_detector(model, os.path.join(img_dir2, file_name.replace("_fail", "")))
        if isinstance(result_p, tuple):
            bbox_results, _ = result_p
            result_p = bbox_results
            bbox_results, _ = result_c
            result_c = bbox_results
        result_above_confidence_num_p = 0
        result_above_confidence_num_c = 0
        result_p = np.concatenate(result_p)
        result_c = np.concatenate(result_c)
        for ir in range(len(result_p)):
            if result_p[ir, 4] > show_score_thr:
                result_above_confidence_num_p = result_above_confidence_num_p + 1
        for ir in range(len(result_c)):
            if result_c[ir, 4] > show_score_thr:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
        if result_above_confidence_num_c == 0:  # can't find any object in clean img
            bb_score = 0
            print('i=', ik)
            print(file_name)
            ik += 1
        else:
            #print("result_above_confidence_num_c", result_above_confidence_num_c)
            bb_score = 1 - min(result_above_confidence_num_c,
                               result_above_confidence_num_p) / result_above_confidence_num_c
        results[file_name] = bb_score
    import json
    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(results, f_obj)
    return results

if __name__ == "__main__":
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    img_file_dir = "../images"
    output_dir="faster_rcnn_output"
    bb_json_name = "whitebox_fasterrcnn_boundingbox_score.json"
    infer(config=config, checkpoint=checkpoint, img_file_dir=img_file_dir + '/',
          output_dir=output_dir, json_name=bb_json_name)
