
from PIL import Image
from torchvision import transforms
import numpy
from mmdet.apis import init_detector, inference_detector
from faster_helper import Helper as FasterHelper
from yolov4_helper import Helper as YoloHelper

from utils.utils import do_detect

def get_yolo_boxes(image_path, yolo_model=YoloHelper().darknet_model):
    img = Image.open(image_path).convert('RGB')
    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img)
    h, w = numpy.array(img).shape[:2]

    boxes = do_detect(yolo_model, img1, 0.5, 0.4, True)
    h, w = numpy.array(img).shape[:2]
    yolo_boxes = [[(box[0] - box[2] / 2.0) * w, (box[1] - box[3] / 2.0) * h,
        (box[0] + box[2] / 2.0) * w, (box[1] + box[3] / 2.0) * h] for box in boxes]

    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    return yolo_boxes

def get_faster_boxes(image_path, faster_model=FasterHelper().model):
    result = inference_detector(faster_model, image_path)
    boxes = []
    for box in result:
        if len(box)>0:
            boxes += box.tolist()

    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
    return boxes

def get_boxes(image_path):
    boxes = get_yolo_boxes(image_path) + get_faster_boxes(image_path)
    boxes = sorted(boxes, key=lambda x:(x[2]-x[0])*(x[3]-x[1])) # sort by area
#    print(boxes)
    return boxes


if __name__ == "__main__":
    get_boxes("./images/991.png")





