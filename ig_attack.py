import math
import random
import asyncio
from mmdet.apis import init_detector, inference_detector
import torch
from torch import nn
import random
from torch import optim
import numpy
from PIL import Image
from torchvision import transforms
import os
from yolov4_helper import Helper as YoLov4Helper
from faster_helper import Helper as FasterHelper
import cv2
from utils.utils import *
from constant import *
import argparse
from integrated_gradient import IntegratedGradients

from bboxes import get_faster_boxes

def create_baseline(img, std=5):
    return img + std


def ig_attack(model_helpers, img_path, save_image_dir, k=100):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float()
    IG = IntegratedGradients(model_helpers)

    t, max_iterations = 0, 1000
    eps = 1
    w = torch.zeros(img.shape).float()+127
    w.requires_grad = True
    success_attack = False
    min_object_num = 1e8
    min_img = img.clone()
    adv_img = img.clone()

    baseline = None
    mask = torch.zeros((min_img.shape[:2]))

    baseline = torch.ones_like(img) * torch.min(img).detach().cpu()
    boxes = get_faster_boxes(img_path)

    ##debug
    #for box in boxes:
    #    model_helpers[0].loss_in_box(img, box)
    #exit(0)
    ##

    add_interval = 60
    max_perturb_num = 500*500*0.015
    max_iterations = add_interval*80
    first_box_add = True
    add_num = 0


    while t<max_iterations:
        if add_num%add_interval==0:
            #baseline = img + (torch.rand(img.shape, device=img.device)-0.5) * 10
            baseline = img * torch.FloatTensor(img.shape, device=img.device).uniform_(0.9, 1.1)

            if len(boxes)==0:
                box = det_bboxes[0]
                #mask_ = IG.get_mask(adv_img.detach(), baseline=baseline.to(adv_img.device))
                k=30
            else:
                box = boxes[0]
                box = [int(item) for item in box]
                k = min(
                        max(int((box[2]-box[0]) * (box[3]-box[1]) * 0.005), 20),
                        200)

            if True:
                while True:
                    mask_ = IG.get_mask(adv_img.detach(), baseline=baseline.to(adv_img.device), box=box)
                    if mask_.sum()==0: 
                        first_box_add = True
                        if len(boxes)>0:
                            boxes=boxes[1:]
                            box = boxes[0]
                        else:
                            det_bboxes = det_bboxes[1:]
                            box = det_bboxes[0]
                            #import pdb; pdb.set_trace()
                    else: break



            #drop_tmp = mask.cpu().numpy()*mask_
            #drop_th = np.sort(drop_tmp.reshape(-1))[k//2]
            #mask[drop_tmp<drop_th]=0

            #tmp_mat = np.ones(mask_.shape) * 1e6
            #tmp_mat[box[1]:box[3], box[0]:box[2]] = 0
            #mask_ =  mask_-tmp_mat# zeros items outside box

            #mask_ = mask_ - mask.numpy()*1e7
            kth = np.sort(mask_.reshape(-1))[::-1][k]
            mask_add = mask_>kth

            mask = mask.cpu().numpy()
            mask_next = (mask+mask_add)>0
            if (mask_next-mask).sum()<30:
                k=30
                mask_add = mask_ - mask*1e7
                kth = np.sort(mask_add.reshape(-1))[::-1][k]
                mask_add = mask_>kth
                mask_next = (mask+mask_add)>0
            mask = mask_next


            if mask.sum()>max_perturb_num: break

            mask = torch.tensor(mask).to(w.device).float()
            print("mask.sum", mask.sum())

        t+=1
        adv_img = img * (1-mask[:,:,None]) + w*mask[:,:,None]
        adv_img = adv_img.to(device)
        attack_loss, object_num = 0, 0
        box_loss, box_object_num = 0, 0
        for helper in model_helpers:
            al, on = helper.attack_loss(adv_img)

            if len(boxes)>0:
                box_al, box_on, det_bboxes = helper.loss_in_box(adv_img, boxes[0])

                box_loss += box_al
                box_object_num += box_on

            attack_loss += al
            object_num += on

        add_interval = 20 if box_loss>10 else 60

        if min_object_num>object_num:
            min_object_num = object_num
            min_img = adv_img.clone()

        if t%5==1: 
            print("t: {}, attack_loss:{}, object_nums:{}, "
                    "len(boxes):{}, box_loss:{}, box_object_num:{}".format(
                t,
                attack_loss, 
                object_num,
                len(boxes),
                box_loss,
                box_object_num
                ))


        if object_num==0:
            success_attack = True
            break
        add_num += 1
        if box_object_num==0 and len(boxes)>0: 
            first_box_add = True
            boxes=boxes[1:]
            add_num = 0
            continue
        if box_loss==0 or len(boxes)==0:
            attack_loss.backward()
        else:
            box_loss.backward()

        w = w - eps * w.grad.sign()
        w = img * (1-mask[:,:,None]) + w*mask[:,:,None]
        w = w.detach().to(mask.device)
        w.requires_grad = True

    try: min_img = min_img.detach().cpu().numpy()
    except Exception: min_img = min_img.numpy()

    if success_attack:
        cv2.imwrite(save_image_dir+"/{}".format(img_path.split("/")[-1]), min_img)
    else:
        cv2.imwrite(save_image_dir+"/{}_fail.png".format(img_path.split("/")[-1].split(".")[0]), min_img)
    return success_attack














if __name__ == "__main__":
    #random.seed(30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_type', type=str, default="grid")
    parser.add_argument('--lines', type=int, default=3)
    parser.add_argument('--box_scale', type=float, default=1.0)
    args = parser.parse_args()
    patch_type = args.patch_type
    lines = args.lines
    box_scale = args.box_scale

    yolov4_helper = YoLov4Helper()
    faster_helper = FasterHelper()
    #model_helpers = [yolov4_helper, faster_helper]
    model_helpers = [faster_helper]
    success_count = 0

    if patch_type == "grid":
        save_image_dir = "images_p_grid_{}x{}_{}".format(lines, lines, box_scale)
    else:
        save_image_dir = "images_p_astroid_{}".format(box_scale)


    save_image_dir = "images_ig"
    os.system("mkdir -p {}".format(save_image_dir))

    images = os.listdir("images")
    random.shuffle(images)

    for i, img_path in enumerate(images):
        img_path_ps = os.listdir(save_image_dir)
        if img_path in img_path_ps:
            success_count+= 1
            continue
        if img_path.replace(".", "_fail.") in img_path_ps: continue
        print("img_path", img_path)
            
        img_path = os.path.join("images", img_path)
        #img_path = os.path.join("images", "2765.png")

        success_attack = ig_attack(model_helpers, img_path, save_image_dir)
        if success_attack: success_count += 1
        print("success: {}/{}".format(success_count, i))

            

