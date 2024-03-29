import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json

import numpy as np
import os
from tool.darknet2pytorch import *
from infer import infer
from tqdm import tqdm
from skimage import measure
from constant import *

def count_detection_score_fasterrcnn(img_file_dir, bb_json_name, output_dir):
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    infer(config=config, checkpoint=checkpoint, img_file_dir=img_file_dir + '/',
          output_dir=output_dir, json_name=bb_json_name)
    return

def count_detection_score_yolov4(selected_path, json_name, output_dir):
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().to(device)

    files = os.listdir(selected_path)
    files.sort()
    bb_score_dict = {}
    for img_name_index in tqdm(range(len(files))):

        img_name = files[img_name_index]
        
        #img_file_dir2 = selected_path.replace('_p', '')  # clean
        img_file_dir2 = selected_path.split("_")[0]
        #img_path0 = os.path.join(img_file_dir2, img_name)
        img_path0 = os.path.join(selected_path.split("_")[0], img_name.split("_")[0]+"png")
        img_path0 = img_path0.replace("_fail", "")
        img0 = Image.open(img_path0).convert('RGB')

        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')


        resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
        ])
        img0 = resize_small(img0)
        img1 = resize_small(img1)

        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0, 0.5, 0.4, True)
        boxes1 = do_detect(darknet_model, img1, 0.5, 0.4, True)

        assert len(boxes0) != 0

        bb_score = 1 - min(len(boxes0), len(boxes1))/len(boxes0)
        bb_score_dict[img_name] = [bb_score, len(boxes1), len(boxes0)]

    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(bb_score_dict, f_obj)

def count_connected_domin_score(max_total_area_rate, selected_path, max_patch_number, json_name, output_dir):

    files = os.listdir(selected_path)
    resize2 = transforms.Compose([
        transforms.ToTensor()])
    files.sort()


    connected_domin_score_dict = {}
    for img_name_index in tqdm(range(len(files))):

        img_name = files[img_name_index]
        #img_path0 = os.path.join(selected_path.replace('_p', ''), img_name)

        img_path0 = os.path.join(selected_path.split("_")[0], img_name.split("_")[0]+"png")
        #img_path0 = img_path0.replace("_fail", "")

        img0 = Image.open(img_path0).convert('RGB')
        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')
        img0_t = resize2(img0).to(device)
        img1_t = resize2(img1).to(device)

        img_minus_t = torch.abs(img0_t - img1_t) # to make all minus positive

        connected_domin_score, total_area_rate, patch_number = \
            connected_domin_detect_and_score(img_minus_t, max_total_area_rate, max_patch_number)
        if patch_number>10: print(img_path0, "patch_num", patch_number)

        if patch_number > max_patch_number:
            connected_domin_score_dict[img_name] = 0.0
            continue

        if patch_number == 0:
            connected_domin_score_dict[img_name] = 0.0
            continue

        if total_area_rate > max_total_area_rate:
            connected_domin_score_dict[img_name] = 0.0
            continue

        connected_domin_score_dict[img_name] = connected_domin_score

    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(connected_domin_score_dict, f_obj)

def connected_domin_detect_and_score(input_img, max_total_area_rate, max_patch_number):
    # detection
    input_img_new = (input_img[0]+input_img[1]+input_img[2])
    #ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    #zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)
    ones = torch.ones(input_img_new.shape).float().to(device)
    zeros = torch.zeros(input_img_new.shape).float().to(device)

    whole_size = input_img_new.shape[0]*input_img_new.shape[1]
    input_map_new = torch.where((input_img_new != 0), ones, zeros)


    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    if max_patch_number > 0:
        if label_max_number > max_patch_number:
            return 0, 0, float(label_max_number)
    if label_max_number == 0:
        return 0, 0, 0

    total_area = torch.sum(input_map_new).item()
    total_area_rate = total_area / whole_size
    
    area_score = 2 - float(total_area_rate/max_total_area_rate)
    return float(area_score), float(total_area_rate), float(label_max_number)

def compute_overall_score(json1, json2, output_dir, output_json):

    with open(os.path.join(output_dir, json1)) as f_obj:
        connected_domin_score_dict = json.load(f_obj)

    with open(os.path.join(output_dir, json2)) as f_obj:
        bbox_score_dict = json.load(f_obj)
    print("len bbox_score", len(bbox_score_dict))
    print(len(connected_domin_score_dict))
    #assert len(bbox_score_dict) == len(connected_domin_score_dict)
    score_sum = 0
    overall_score = {}
    APP, BBR_0, BBR_1, SR = 0, 0, 0, 0
    for (k, _) in bbox_score_dict.items():
        if not k in connected_domin_score_dict: continue
        overall_score[k] = connected_domin_score_dict[k] * bbox_score_dict[k][0]
        score_sum += connected_domin_score_dict[k] * bbox_score_dict[k][0]
        APP += (2-connected_domin_score_dict[k])/50
        BBR_0 += bbox_score_dict[k][1]
        BBR_1 += bbox_score_dict[k][2]
        if bbox_score_dict[k][1]==0: SR+=1

    print('Overall score: ', score_sum)
    print("BBR", BBR_0/BBR_1)
    print("SR", SR/len(bbox_score_dict.keys()))
    print("APP", APP/len(bbox_score_dict.keys()))
    print('Saving into {}...'.format(output_json))
    with open(os.path.join(output_dir, output_json), 'w') as f_obj:
        json.dump(
                [
                    score_sum,
                    BBR_0/BBR_1,
                    SR/len(bbox_score_dict.keys()),
                    APP/len(bbox_score_dict.keys())
                ],  f_obj)



if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
    max_patch_number = 10000

    selected_paths = ["../"+selected_path for selected_path in os.listdir("..") if "images_p" in selected_path]

    adv_dir = "./images_integrated_grad/faster/"
    adv_dir = "./images_grad_input/faster/"

    adv_dir = "./images_grad/yolo/"
    adv_dir = "./images_random/yolo/"
    for model in ['faster', 'yolo']:
        #for attack_type in ['random', 'grad', 'grad_input', 'integrated_grad']:
        for attack_type in ['images_integrated_grad_ones/', 'images_integrated_grad_zeros/','images_integrated_grad_half/', 'images_integrated_grad_rand/', 'images_integrated_grad_near/']:

            #adv_dir = "./images_{}/{}/".format(attack_type, model)
            #selected_paths = [os.path.join(adv_dir, str(int(i)))+"/" for i in range(1, 17, 2)]

            adv_dir = "./{}/{}/".format(attack_type, model)
            selected_paths = [os.path.join(adv_dir, str(int(i)))+"/" for i in [15]]

            for selected_path in selected_paths:
                assert len(os.listdir(selected_path))==100, selected_path
                output_dir = selected_path.replace("./images", "./output_data")
            
                os.system("mkdir -p " + output_dir)
                # compute_connected_domin_score
                cd_json_name = 'connected_domin_score.json'
                if not os.path.exists(os.path.join(output_dir, cd_json_name)): 
                    count_connected_domin_score(MAX_TOTAL_AREA_RATE, selected_path, max_patch_number, cd_json_name, output_dir)
            
                # compute_boundingbox_score
                bb_json_name = 'whitebox_yolo_boundingbox_score.json'
                whitebox_yolo_result = 'whitebox_yolo_overall_score.json'
                if not os.path.exists(os.path.join(output_dir, bb_json_name)):
                    count_detection_score_yolov4(selected_path, bb_json_name, output_dir)
                compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_yolo_result)
            
                bb_json_name = 'whitebox_fasterrcnn_boundingbox_score.json'
                whitebox_fasterrcnn_result = 'whitebox_fasterrcnn_overall_score.json'
                if not os.path.exists(os.path.join(output_dir, bb_json_name)):
                    count_detection_score_fasterrcnn(selected_path, bb_json_name, output_dir)    
                compute_overall_score(cd_json_name, bb_json_name, output_dir, whitebox_fasterrcnn_result)
        
