# -*- coding:utf-8 -*-
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['SimHei'],
    })

#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False



def get_results(outdir):
    faster_res, yolo_res = [], []
    for i in range(1,17, 2):
        try:
            json_path = os.path.join(outdir, str(i),  "whitebox_fasterrcnn_overall_score.json")
            with open(json_path, "r") as infile:
                faster_res += [json.load(infile)]
            json_path = os.path.join(outdir, str(i),  "whitebox_yolo_overall_score.json")
            with open(json_path, "r") as infile:
                yolo_res += [json.load(infile)]
        except Exception: break
    return faster_res, yolo_res

#def get_linestype(i):
#    styles = ['-', '--', '-.', ':', 'None', ' ', '']
#    return styles[i]

def get_color(i):
    colors = ['blue', 'cyan', 'greenyellow', 'purple']
    return colors[i]



if __name__ == "__main__":
    output_dir = "output_data_random/yolo/"
    faster_res, yolo_res = get_results(output_dir)
    output_dir = "output_data_grad/yolo/"
    faster_res, yolo_res = get_results(output_dir)

    names = ['random', 'grad', 'grad_input', 'integrated_grad']
    model, task = "faster", "score"
    for model in ['faster', 'yolo']:
        for task in ['score', 'app']:
            filter_max = 17
            x = np.arange(1, filter_max, 2)


            for idx, name in enumerate(names):
                scores, apps = [], []

                output_dir = "output_data_{}/{}/".format(name, model)
                faster_res, yolo_res = get_results(output_dir)
                res = faster_res if model=="faster" else yolo_res
                scores+=[int(item[0]*10) for item in res]
                apps+=[item[-1]*(500**2) for item in res]

                scores = scores[:len(x)]
                apps = apps[:len(x)]
                data = scores if task=="score" else apps
                #print(f"{model}, {task}, {name}", data)
                #print(f"{model}, {task}, {name}", res[-1])

                # plot
                plt.plot(x, np.array(data), linestyle='--', label=name, color=get_color(idx)) 

            plt.xticks(x, x, rotation ='horizontal')
            plt.legend()

            plt.xlabel(u"池化核的大小")
            plt.ylabel(u"平均扰动像素数目" if task=='app' else u"总得分")

            plt.savefig("{}_{}.png".format(task, model))
            plt.clf()
