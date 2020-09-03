## Introduction

This is a project for the Tianchi competition: adversarial attack for universal object detection. Here is the url: https://tianchi.aliyun.com/competition/entrance/531806/information. We obtain the third in this contest.

## Data preparation and model checkpoint.

- Download 1000 pictures needed for the competition on the official [website](https://tianchi.aliyun.com/competition/entrance/531806/information)
- You can get data (`images.zip`) and the definition, weight and evaluation code of two white box models (`eval_code.zip`). We use yolov4 and faster_rcnn as whitebox models.
- Create two new folders, `images` and `models`, Unzip `images.zip` to `images`, and move all checkpoint and config files to models.

## Requirements

This code is based on pytorch. Some basic dependencies are recorded in `requirements.txt`

- torch
- torchvision
- pillow
- numpy
- tqdm
- scipy
- scikit-image
 
You can run yolov4 now if all above requirements are satisfied.

Another faster rcnn model is implemented based on mmdetection. So, ensure that the mmdetection library has been installed and can be run on your machine. You can refer install guide of mmdetection to [github](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md)

After installation, put the mmdetection directory into `eval_code/` below. Alternatively, it is optional that using [docker](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) provided by mmdetection.

## Usage

Unzip `eval_code.zip`，move and unzip `images.zip` to images`, ensure the following structure:

```
|--images
    |-- XXX.png
    |-- XXX.png
    |-- XXX.png
    …
    |-- XXX.png
```

Move all checkpoints and config files to models as:

```
|--models
    |-- faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    |-- yolov4.cfg
    |-- yolov4.weights
```

## Run Attack algorithm

```bash
python attack.py --patch_type grid --lines 3 --box_scale 1.0
python attack.py --patch_type grid --lines 2 --box_scale 1.0
python attack.py --patch_type grid --lines 1 --box_scale 1.0


python attack.py --patch_type astroid
```

## Run ensemble algorithm
```bash
python ensemble.py
```
