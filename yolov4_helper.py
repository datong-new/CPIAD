import torch
from torch.nn import functional as F
from torch import nn
import cv2
from torchvision import transforms
from PIL import ImageFile
from PIL import Image
from tool.darknet2pytorch import *
import numpy
from constant import *

def resize(img:torch.tensor):
    # resize
    img_pil = img.round().byte().detach().cpu().numpy()
    img_pil = Image.fromarray(img_pil.astype('uint8'), 'RGB')
    resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
        ])
    img_pil = resize_small(img_pil)
    img_pil = numpy.array(img_pil)
    img_pil = torch.from_numpy(img_pil).float().to(device)

    img = img.permute(2,0,1)
    img = nn.functional.interpolate(img.unsqueeze(0), size=(608, 608),mode="bilinear",  align_corners=False)
    img = img.squeeze(0).permute(1,2,0)
    img = img + (img_pil-img.detach())
    return img

def _input_transform(img:torch.tensor):
    ## bgr2rgb
    new_img = torch.zeros(img.shape).to(device)
    new_img[:,:,0] = new_img[:,:,0]+img[:,:,2]
    new_img[:,:,1] = new_img[:,:,1]+img[:,:,1]
    new_img[:,:,2] = new_img[:,:,2]+img[:,:,0]
    img = new_img

    # reisze
    img = resize(img)

    img = img.permute(2,0,1).contiguous()
    #img = img.float().div(255.0)
    img = img.float()/255.0
    img = img.unsqueeze(0)
    return img

def transforms_test(img_path="../images/991.png"):
    img1 = Image.open(img_path).convert('RGB')
    resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
        ])
    img_pil = numpy.array(img1)
    img1 = resize_small(img1)

    width = img1.width
    height = img1.height
    img1 = torch.ByteTensor(torch.ByteStorage.from_buffer(img1.tobytes()))
    img1 = img1.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img1 = img1.view(1, 3, height, width)
    img1 = img1.float().div(255.0).to(device)

    img2 = cv2.imread(img_path)
    img2 = torch.from_numpy(img2).float()
    img2 = _input_transform(img2).to(device)
    print((img1-img2).sum())
    assert (img1-img2).sum()==0

class Helper():
    def __init__(self):
        cfgfile = "models/yolov4.cfg"
        weightfile = "models/yolov4.weights"
        self.darknet_model = Darknet(cfgfile)
        self.darknet_model.load_weights(weightfile)
        #self.darknet_model = self.darknet_model.train().to(device)
        self.darknet_model = self.darknet_model.eval().to(device)

    def input_transforms(self, img:torch.tensor):
        img = _input_transform(img)

    def get_cls_scores(self, img:torch.tensor):
        img = _input_transform(img).to(device)
        output = self.darknet_model(img)
        self.features = self.darknet_model.features
        scores = []
        for item in output:
            h, w = item.shape[-2], item.shape[-1]
            item = item.reshape(-1, 5+80, h*w).permute(1,0,2).reshape(5+80, -1)
            scores += [item[4, :].sigmoid()]

        return scores

    def loss_in_box(self, img, box):
        img = _input_transform(img).to(device)
        output = self.darknet_model(img)

        import pdb;pdb.set_trace()

        img_h, img_w = img.shape[-2:]
        out_h, out_w = output.shape[-2:]
        scale_h, scale_w = out_h*1.0/img_h, out_w*1.0/img_w
        box = [box[0]*scale_w, box[1]*scale_h, box[2]*scale_w, box[3]*scale_h]
        output = output[:,:,box[1]:box[3], box[0]:box[2]]






    def attack_loss(self, img, t=0.45):
        img = img.to(device)
        scores = self.get_cls_scores(img)
        thresh_loss = 0
        objects_num = 0
        for score in scores:
            objects_num += (score>0.5).sum().item()
            mask = score>0.45
            score = score *mask
            if mask.sum()!=0: thresh_loss += (score.sum() / mask.sum())
        return thresh_loss, objects_num

if __name__ == "__main__":
    transforms_test()

    img_path = "../images/991.png"
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float()
    helper = Helper()
    attack_loss = helper.attack_loss(img)
    print(attack_loss)
