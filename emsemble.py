import os
import json

image_dirs = [img_dir for img_dir in os.listdir(".") if "images_p" in img_dir]
json_dirs = [image_dir.replace("images", "output_data") for image_dir in image_dirs]


image_paths = os.listdir("images")
scores = {}
for i, json_dir in enumerate(json_dirs):

    if not os.path.exists(os.path.join(json_dir, "whitebox_yolo_overall_score.json")): continue
    with open(os.path.join(json_dir, "whitebox_yolo_overall_score.json"), "r") as infile:
        yolo_scores = json.load(infile)

    if not os.path.exists(os.path.join(json_dir, "whitebox_fasterrcnn_overall_score.json")): continue
    with open(os.path.join(json_dir, "whitebox_fasterrcnn_overall_score.json"), "r") as infile:
        faster_scores = json.load(infile)

    new_yolo_scores = {}
    for k, v in yolo_scores.items():
        new_yolo_scores[k.replace("_fail", "")] = v
    yolo_scores = new_yolo_scores

    new_faster_scores = {}
    for k, v in faster_scores.items():
        new_faster_scores[k.replace("_fail", "")] = v
    faster_scores = new_faster_scores

    for k in yolo_scores.keys():
        if not k in scores: scores[k] = [yolo_scores[k]+faster_scores[k], image_dirs[i]]
        else:
            if scores[k][0]<yolo_scores[k]+faster_scores[k]:
                scores[k] = [yolo_scores[k]+faster_scores[k], image_dirs[i]]

#print(scores)

selected_images_dir = "select1000_new_p/"
score_sum = 0
fail_count = 0

for k in scores.keys():
    img_dir = scores[k][1]
    if scores[k][0]==0: 
        zeros+=1

    k = k.replace("_fail", "")
    score_sum += scores[k][0]

    if os.path.exists(os.path.join(img_dir, k)):
        os.system("cp {} {}".format(os.path.join(img_dir, k), selected_images_dir))
    else:
        fail_count += 1
        os.system("cp {} {}".format(os.path.join(img_dir, k.replace(".", "_fail.")), 
            os.path.join(selected_images_dir, k)))
print("score sum:", score_sum)
