import os
import numpy as np

def count(img_dir):
    nums = []
    for img_path in os.listdir(img_dir):
        num = img_path.split("_")[-1].split(".")[0]
        nums+=[int(num)]
    nums = np.array(nums)
    return nums.max(), int(nums.mean()), nums.min()


if __name__=="__main__":
    
    for model in ['faster', 'yolo']:
        for attack_type in ['random', 'grad', 'grad_input', 'integrated_grad']:
            img_dir = "./images_{}/{}/15/".format(attack_type, model)
            max_,mean,min_ = count(img_dir)
            app = mean / (500*500)
            print(f"{model}, {attack_type}: {mean}& {max_} & {min_} & {app}")
