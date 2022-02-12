import math
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
    print("len(boxes)", len(boxes))



    while t<max_iterations:
        if t%50==0:
            while True and len(boxes)>0:
                mask_ = IG.get_mask(adv_img.detach(), baseline=baseline.to(adv_img.device), box=boxes[0])
                if mask_.sum()==0: 
                    boxes=boxes[1:]
                else: break

            mask_ = mask_ - mask.numpy()*1e7
            kth = np.sort(mask_.reshape(-1))[::-1][k]
            mask_ = mask_>kth

            mask = mask.cpu().numpy()
            if (mask+mask_).sum()<500*500*0.01: 
                mask = (mask+mask_)>0
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
                box_al, box_on = helper.loss_in_box(adv_img, boxes[0])

                box_loss += box_al
                box_object_num += box_on

            attack_loss += al
            object_num += on

        print("len(boxes)", len(boxes))


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
        if box_object_num==0 and len(boxes)>0: 
            boxes=boxes[1:]
            continue

        attack_loss.backward()
        #box_loss.backward()
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
    random.seed(30)
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
    os.system("mkdir -p {}".format(save_image_dir))


    for i, img_path in enumerate(os.listdir("images")):
        img_path_ps = os.listdir(save_image_dir)
        if img_path in img_path_ps:
            success_count+= 1
            continue
        if img_path.replace(".", "_fail.") in img_path_ps: continue
        print("img_path", img_path)
            
        img_path = os.path.join("images", img_path)

        success_attack = ig_attack(model_helpers, img_path, save_image_dir)
        if success_attack: success_count += 1
        print("success: {}/{}".format(success_count, i))

            

