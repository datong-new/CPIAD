import torch
from torch import nn
import cv2
import numpy as np
from generative_net import get_network
from yolov4_helper import Helper as YoLov4Helper
from faster_helper import Helper as FasterHelper
import os
from constant import *

def get_patch_img(patch_positions, patch, img, patch_size=20):
    mask = torch.zeros(img.shape).to(device) # batch x 3 x h x w
    patch_positions = patch_positions.reshape(-1, 2)
    for i, position in enumerate(patch_positions):
        x, y = int(position[0].item()), int(position[1].item())
        x = np.clip(x, patch_size//2, (500-patch_size//2))
        y = np.clip(y, patch_size//2, (500-patch_size//2))
        mask[:, :, x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2] = 1

    patch = torch.clamp(patch, 0, 255)

    patch_img = img * (1-mask) + mask * patch
    cv2.imwrite("test.png", patch_img.squeeze().permute(1,2,0).detach().cpu().numpy())
    return patch_img

if __name__ == "__main__":
    net = get_network()
    net = net.to(device)
    yolov4_helper = YoLov4Helper()
    faster_helper = FasterHelper()
    #model_helpers = [yolov4_helper, faster_helper]
    #model_helpers = [yolov4_helper]
    model_helpers = [faster_helper]

    img_paths = [os.path.join("../images", img_path) for img_path in os.listdir("../images")]
    epoch = 100
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    eps = 1
    patch_size = 20
    patch_num = 10

    ave_pool = nn.AvgPool2d(patch_size, stride=patch_size)

    for e in range(epoch):
        loss_sum, objects_sum = 0, 0
        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0).to(device)
            patch_positions, patch = net(img)
            """
            patch_regularize_loss = (patch_positions[patch_positions>500]**2).sum() + \
                    (patch_positions[patch_positions<0]**2).sum() + \
                    (patch[patch<0]**2).sum() + \
                    (patch[patch>255]**2).sum()
            """
            patch_regularize_loss = 0


            patch_img = get_patch_img(patch_positions, patch, img)
            attack_loss = 0
            objects_num = 0

            patch_img = patch_img.squeeze().permute(1,2,0)
            positions_gt = []

            patch_img_1 = patch_img.detach()
            patch_img_1.requires_grad = True
            for helper in model_helpers:
                al, on = helper.attack_loss(patch_img_1)
                attack_loss += al
                objects_num += on

                
            attack_loss.backward()
            grad = patch_img_1.grad
            grad = grad.permute(2,0,1).unsqueeze(0)
            # find max grad positions
            grad = ave_pool(grad)
            grad = grad.sum(dim=1)
            grad_1 = grad
            grad = torch.abs(grad.squeeze().view(-1))
            grad_topk = grad.topk(patch_num)
            for item in grad_topk[1]:
                mod_value = 500 // patch_size
                x = item.item() % mod_value 
                y = item.item() // mod_value

                x *= patch_size
                y *= patch_size
                positions_gt += [y, x]
                #print("x:{}, y:{}".format(x, y))
                #print("grad 0", grad_1[0, y//20, x//20])
                #print("grad 1", grad[item])

            patch_img_1 = patch_img_1 - eps * patch_img_1.grad.sign()

            generative_loss = (torch.abs(patch_positions.squeeze() - torch.tensor(positions_gt, dtype=torch.float).to(device))).mean() + \
                    (torch.abs(patch_img-patch_img_1)).mean()
            print("epoch:{}, iterations:{}, attack_loss:{}, objects_num:{}, generative_loss:{}".format(e, i, 
                attack_loss.item(), objects_num, generative_loss))
            loss_sum += attack_loss.item()
            objects_sum += objects_num

            optimizer.zero_grad()
            loss = patch_regularize_loss + generative_loss
            loss.backward()
            optimizer.step()
        with open("save_models_10x20x20/log.txt", "a") as infile:
            infile.write("epoch:{}, loss_sum:{}, objects_sum:{}\n".format(e, loss_sum, objects_sum))

        if (e+1) % 5 == 0:
            torch.save(net.state_dict(), "save_models/model_{}.pt".format(e+1))
