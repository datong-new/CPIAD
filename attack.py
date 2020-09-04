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

def create_astroid_mask(darknet_model, faster_model, image_path, box_scale, shape=(500, 500)):
    mask = torch.zeros(*shape, 3)
    """
    img = Image.open(img_path).convert('RGB')
    resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(darknet_model, img1, 0.5, 0.4, True)
    h, w = numpy.array(img).shape[:2]
    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]
    boxes = yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    """

    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()

    boxes = [box[:4] for box in boxes if box[-1]>0.3]
    #boxes += yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)
    num = 0
    for _, (x1, y1, x2, y2) in enumerate(grids):
        if num>9: break
        x1 = int(np.clip(x1, 0, 499))
        x2 = int(np.clip(x2, 0, 499))

        y1 = int(np.clip(y1, 0, 499))
        y2 = int(np.clip(y2, 0, 499))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2

        # shrink box
        box_h, box_w = int((y2-y1)*box_scale), int((x2-x1)*box_scale)
        y11 = y_middle-box_h//2
        y22 = y_middle+box_h//2
        x11 = x_middle-box_w//2
        x22 = x_middle+box_w//2


        cross_line_x_len = x_middle-x11
        cross_line_y_len = y_middle-y11
        cross_line_len = max(y_middle-y11, x_middle-x11)
        y_step, x_step = cross_line_y_len/cross_line_len, cross_line_x_len/cross_line_len

        tmp_mask = torch.zeros(mask.shape)
        tmp_mask[y_middle, x11:x22, :] = 1
        tmp_mask[y11:y22, x_middle, :] = 1
        for i in range(1, cross_line_len):
            tmp_mask[y_middle-int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle-int(i*x_step), :] = 1
            tmp_mask[y_middle-int(i*y_step), x_middle+int(i*x_step), :] = 1
            tmp_mask[y_middle+int(i*y_step), x_middle+int(i*x_step), :] = 1
        before_area = tmp_mask.sum()
        after_area = (tmp_mask*(1-visited_mask)).sum()
        if float(after_area) / float(before_area) < 0.5:
            continue

        if (mask+tmp_mask).sum()>5000*3: break
        num += 1
        mask = mask + tmp_mask
        visited_mask[y1:y2, x1:x2, :] = 1
    print("mask sum", mask.sum())
    return mask


def create_grid_mask(darknet_model, faster_model, image_path, lines=3, box_scale=1.0, shape=(500, 500)):
    mask = torch.zeros(*shape, 3)
    """
    img = Image.open(img_path).convert('RGB')
    resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(darknet_model, img1, 0.5, 0.4, True)
    h, w = numpy.array(img).shape[:2]
    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h, 
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]
    boxes = yolo_boxes
    grids = boxes
    """

    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()

    boxes = [box[:4] for box in boxes if box[-1]>0.3]
    #boxes += yolo_boxes
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    grids = boxes
    mask = torch.zeros(*shape, 3)
    visited_mask = torch.zeros(*shape, 3)

    for x1, y1, x2, y2 in grids:
        x1 = int(np.clip(x1, 0, 499))
        x2 = int(np.clip(x2, 0, 499))

        y1 = int(np.clip(y1, 0, 499))
        y2 = int(np.clip(y2, 0, 499))

        print("x1, y1, x2, y2", x1, y1, x2, y2)
        y_middle = (y1+y2)//2
        x_middle = (x1+x2)//2
        # shrink box
        box_h, box_w = int((y2-y1)*box_scale), int((x2-x1)*box_scale)
        y11 = y_middle-box_h//2
        y22 = y_middle+box_h//2
        x11 = x_middle-box_w//2
        x22 = x_middle+box_w//2

        min_interval = 32
        if lines == 0:
            min_interval = 0

        y_interval = max(min_interval, (y2-y1)//(lines+1))
        x_interval = max(min_interval, (x2-x1)//(lines+1))
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if y1+i*y_interval>y2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[np.clip(y1+i*y_interval, 0, 499), x11:x22, :]=1
            before_area = tmp_mask.sum()
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.5: continue
            mask = mask + tmp_mask
        for i in range(1, lines+1):
            if mask.sum()>4500*3: break
            if x1+i*x_interval>x2: break
            tmp_mask = torch.zeros(*shape, 3)
            tmp_mask[y11:y22, np.clip(x1+i*x_interval, 0, 499), :]=1
            before_area = tmp_mask.sum()
            after_area = (tmp_mask*(1-visited_mask)).sum()
            if float(after_area) / float(before_area) < 0.5: continue
            mask = mask + tmp_mask

        visited_mask[y1:y2, x1:x2, :] = 1

    print("mask sum", mask.sum())
    return mask

def get_delta(w):
    w = torch.clamp(w, 0, 255)
    #return (1+(1-torch.exp(w))/(1+torch.exp(w))) * 127 # [0, 2*127] = [0, 254]
    return w

async def get_attack_loss(helper, img):
    al, on = await helper.attack_loss(img)
    return al, on

def specific_attack(model_helpers, img_path, mask, save_image_dir):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float()

    t, max_iterations = 0, 600
    stop_loss = 1e-6
    eps = 1
    w = torch.zeros(img.shape).float()+127
    w.requires_grad = True
    success_attack = False
    patch_size = 70
    patch_num = 10
    min_object_num = 1000
    min_img = img
    grads = 0
    loop = asyncio.get_event_loop()
    while t<max_iterations:
        t+=1

        # check connectivity
        patch_connecticity = torch.abs(get_delta(w)-img).sum(-1)
        patch_connecticity = (patch_connecticity==0)
        patch = get_delta(w)
        patch[patch_connecticity] += 1

        patch_img = img * (1-mask) + patch*mask
        patch_img = patch_img.to(device)

        attack_loss = 0
        object_nums = 0
        # https://xubiubiu.com/2019/06/12/python3-%E8%8E%B7%E5%8F%96%E5%8D%8F%E7%A8%8B%E7%9A%84%E8%BF%94%E5%9B%9E%E5%80%BC/
        tasks = [
                get_attack_loss(model_helpers[0], patch_img),
                get_attack_loss(model_helpers[1], patch_img),
                ]
        res = loop.run_until_complete(asyncio.gather(*tasks))
        for al, on in res:
            attack_loss += al
            object_nums += on

        if min_object_num>object_nums:
            min_object_num = object_nums
            min_img = patch_img
        if object_nums==0:
            success_attack = True
            break
        if t%20==0: print("t: {}, attack_loss:{}, object_nums:{}".format(t, attack_loss, object_nums))
        attack_loss.backward()
        #grads = grads + w.grad / (torch.abs(w.grad)).sum()
        w = w - eps * w.grad.sign()
        w = w.detach()
        w.requires_grad = True

    min_img = min_img.detach().cpu().numpy()
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
    model_helpers = [yolov4_helper, faster_helper]
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
        if patch_type=="grid":
            mask = create_grid_mask(yolov4_helper.darknet_model, faster_helper.model, img_path, lines, box_scale)
        else:
            mask = create_astroid_mask(yolov4_helper.darknet_model, faster_helper.model, img_path, box_scale)
        success_attack = specific_attack(model_helpers, img_path, mask, save_image_dir)
        if success_attack: success_count += 1
        print("success: {}/{}".format(success_count, i))

            


