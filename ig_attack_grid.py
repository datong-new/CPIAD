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

def create_adv_baseline(img, helpers, iterations=10, eps=1):
    img = img.clone().detach()
    m = 0
    for i in range(iterations):
        img.requires_grad = True
        attack_loss = 0
        for helper in model_helpers:
            al, on = helper.attack_loss(img)
            attack_loss += al
        #print("attck_loss", attack_loss)
        if attack_loss<=0: break
        attack_loss.backward()
        m = 0.8 * m + 0.2 * img.grad
        img = img-eps*torch.sign(m)
        img = img.detach()
    return img

def get_k(attack_loss):
    if attack_loss<1: 
        k=30
    elif attack_loss<5: 
        k=50
    elif attack_loss<10:
        k=100
    else: k=200

    return k

def get_k_by_num(num):
    if num<10:
        k = 30
    elif num<20:
        k=100
    else:
        k=200
    return k


def ig_attack(model_helpers, img_path, save_image_dir, k=100):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float().cuda()
    IG = IntegratedGradients(model_helpers)

    t, max_iterations = 0, 1000
    eps = 1
    w = torch.zeros(img.shape, device='cuda').float()+127
    w.requires_grad = True
    success_attack = False
    min_object_num = 1e8
    min_mask_sum = 1e8
    min_img = img.clone()
    adv_img = img.clone()

    baseline = None
    mask = np.zeros((min_img.shape[:2]))

    baseline = torch.ones_like(img) * torch.min(img).detach().cpu()
    boxes = get_faster_boxes(img_path)

    mask_in_boxes = np.zeros(img.shape[:-1])
    for box in boxes:
        box = [int(item) for item in box]
        mask_in_boxes[box[1]:box[3], box[0]:box[2]] = 1

    add_interval = 60
    max_perturb_num = 500*500*0.02
    max_iterations = 3000
    first_box_add = True
    add_num = 0
    attack_loss = 1000
    box_loss = 0
    object_num = 1000

    def get_topk(mask_, k=20):
        kth = np.sort(mask_.reshape(-1))[::-1][k]
        mask = mask_>kth
        return mask

    def drop_mask(mask, k=100):
        mask_reshape = mask.reshape(-1)

        kth = np.sort(mask_reshape[mask_reshape!=0])[k-1]
        mask = (mask>=kth) * (mask!=0)
        return mask

    def make_grid(mask, size=3, stride=2):
        mask_copy = mask.copy()
        for i in range(1, size//2+1):
            mask_copy[i:, :] += mask[:-i, :]
            mask_copy[:-i, :] += mask[i:, :]

            mask_copy[:, i:] += mask[:, :-i]
            mask_copy[:, :-i] += mask[:, i:]
        return mask_copy


    #baseline = create_adv_baseline(adv_img, model_helpers)
    while t<max_iterations:
        if t%add_interval==0 or object_num<2:
            baseline = create_adv_baseline(adv_img, model_helpers)
            mask_ = IG.get_mask(adv_img.detach(), baseline=baseline.detach().to(adv_img.device))


            if object_num<2:
                mask = drop_mask(mask*mask_, k=50)
            elif object_num>5:
                #k = get_k(attack_loss)
                k = get_k_by_num(object_num)
                mask = mask + get_topk(mask_*mask_in_boxes, k=k)

            #mask_grid = make_grid(mask)
            mask_grid = mask
            mask_grid = torch.tensor(mask_grid).to(w.device).float()
            print("mask.sum()", mask_grid.sum())

        t+=1
        adv_img = img * (1-mask_grid[:,:,None]) + w*mask_grid[:,:,None]
        adv_img = adv_img.to(device)
        attack_loss, object_num = 0, 0
        box_loss, box_object_num = 0, 0
        for helper in model_helpers:
            al, on = helper.attack_loss(adv_img)

            attack_loss += al
            object_num += on

        if t%5==1: 
            print("t: {}, attack_loss:{}, object_nums:{}, success_attack:{},"
                    "min_mask_sum:{}".format(
                t,
                attack_loss, 
                object_num,
                success_attack, 
                min_mask_sum
                ))
        if min_object_num>object_num or (min_object_num==object_num and min_mask_sum>mask_grid.sum()):
            min_object_num=object_num
            min_img = adv_img.clone()
            if min_object_num==object_num and min_mask_sum>mask_grid.sum():
                min_mask_sum = mask_grid.sum()


        if object_num==0:
            success_attack = True
            break
        attack_loss.backward()

        #m = 0.5 * m + 0.5 * w.grad
        m = w.grad

        w = w - eps * m.sign()
        w = w.detach().to(adv_img.device)
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
    parser.add_argument('--save_image_dir', type=str, default=None)
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

    if args.save_image_dir is not None:
        save_image_dir = args.save_image_dir

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

            
