import torch
import numpy as np
import cv2
import mmcv
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector
from mmdet.core import bbox2result, bbox2roi
from torch import nn
from constant import device1 as device

mean = torch.tensor([123.675, 116.28 , 103.53 ], dtype=torch.float)
std = torch.tensor([58.395, 57.12 , 57.375] ,dtype=torch.float)

class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def resize(img:torch.tensor):
    img_np = img.round().byte().detach().cpu().numpy()
    img_np = cv2.resize(img_np, (800,800), interpolation=cv2.INTER_LINEAR)
    img_np = torch.from_numpy(img_np).float().to(device)

    img = img.permute(2,0,1)
    img = nn.functional.interpolate(img.unsqueeze(0), size=(800, 800),mode="bilinear",  align_corners=False)
    img = img.squeeze(0).permute(1,2,0)
    img = img + (img_np-img.detach())
    return img

def flip(img:torch.tensor):
    return img

def normalize(img:torch.tensor):
    ## bgr2rgb
    new_img = torch.zeros(img.shape).to(device)
    new_img[:,:,0] = new_img[:,:,0]+img[:,:,2]
    new_img[:,:,1] = new_img[:,:,1]+img[:,:,1]
    new_img[:,:,2] = new_img[:,:,2]+img[:,:,0]
    img = new_img
    img = (img-mean.to(device))/std.to(device)
    return img

def pad(img:torch.tensor):
    return img

def _input_transforms(img:torch.tensor):
    img = resize(img)
    img = flip(img)
    img = normalize(img)
    img = pad(img)
    img = img.permute(2,0,1)
    return img

def transforms_test(img_path="../images/991.png"):
    helper = Helper()
    helper.data_init(img_path)
    img1 = helper.data['img'][0]


    img2 = cv2.imread(img_path)
    img2 = torch.from_numpy(img2).float()
    img2 = _input_transforms(img2).to(device)

    print("img1-img2", img1-img2)
    print("img1-img2", (img1-img2).sum())
    assert (img1-img2).sum()==0

class Helper():
    def __init__(self):
        config = './mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        checkpoint = './models/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.model = init_detector(config, checkpoint, device=device)
        self.model.eval()
        self.get_img_metas()

    def input_transforms(self, img:torch.tensor):
        return _input_transforms(img).to(device)

    def data_init(self, img_path):
        # its main purpose is to obtain the img_metas.
        # besides, we compare the image we transform with self.data['img'].
        cfg = self.model.cfg
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        self.data = dict(img=img_path)
        self.data = test_pipeline(self.data)
        self.data = collate([self.data], samples_per_gpu=1)
        self.data = scatter(self.data, [device])[0]

    def get_img_metas(self):
        self.data_init("../images/991.png")
        self.img_metas = self.data['img_metas'][0]

    def get_rpn_cls_scores(self, img):
        feat = self.model.extract_feat(img)
        cls_scores, dets = self.model.rpn_head(feat)
        print(len(cls_scores))
        print(len(dets))
        print("cls_scores.shape", cls_scores[0].shape)
        print("dets.shape", dets[0].shape)
        print("cls_scores.shape", cls_scores[1].shape)
        print("dets.shape", dets[1].shape)
        print("cls_scores.shape", cls_scores[2].shape)
        print("dets.shape", dets[2].shape)
        for i in range(len(cls_scores)):
            cls_scores[i] = cls_scores[i].sigmoid()
        return cls_scores

    def get_detector_cls_sorces(self, img, img_metas, proposals=None, rescale=True):


        """Test without augmentation."""
        assert self.model.with_bbox, 'Bbox head must be implemented.'
        x = self.model.extract_feat(img)
        if proposals is None:
            proposal_list = self.model.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        assert self.model.roi_head.with_bbox

        ms_scores = []
        rois = bbox2roi(proposal_list)
        for i in range(self.model.roi_head.num_stages):
            bbox_results = self.model.roi_head._bbox_forward(i, x, rois)
            ms_scores.append(bbox_results['cls_score'])
            if i < self.model.roi_head.num_stages - 1:
                bbox_label = bbox_results['cls_score'][:, :-1].argmax(dim=1)
                rois = self.model.roi_head.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])

        cls_scores = sum(ms_scores) / self.model.roi_head.num_stages

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_boxes, det_scores = self.model.roi_head.bbox_head[-1].get_bboxes(
            rois,
            cls_scores,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=None)
        return det_scores[:, :-1] # background is the last class and exclude it


    async def attack_loss(self, img, t=0.25):
        assert self.img_metas is not None
        img = img.to(device)
        img = self.input_transforms(img).unsqueeze(0)
        #rpn_scores, rpn_dets = self.get_rpn_cls_scores(img)
        scores = self.get_detector_cls_sorces(img, self.img_metas)
        objects_num = (scores>0.3).sum().item()
        
        mask = scores>t
        scores = scores * mask

        thresh_loss = scores.sum()
        return thresh_loss, 0


if __name__ == "__main__":
    img_path = "../images/991.png"
    #transforms_test(img_path)

    helper = Helper()

    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float().to(device)
    faster_loss = helper.attack_loss(img)
    print("faster_loss", faster_loss)
    exit(0)

