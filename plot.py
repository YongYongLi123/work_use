import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


def distance_scatter_plot(type,dim,data):
    x = np.linspace(1,len(data),len(data))
    y =data
    plt.figure('distance')
    plt.scatter(x,y,s=1)
    plt.savefig('./dataset_16568_dali/figure/散点图_{}_{}.png'.format(type,dim),dpi=500)
    plt.show()

def fit_normal_distribution(type,dim,data):
    x = data
    sns.set_palette('hls') # 设置图的颜色,使用hls颜色空间
    sns.distplot(x, color="b", bins=10,kde=True)
    plt.savefig('./dataset_16568_dali/figure/直方图_{}_{}.png'.format(type,dim),dpi=500)
    plt.show()

def get_distance(data_dir, data_list):
    data_dict = {
                "person":{"length":[],"width":[],"height":[]},
                "pushing":{"length":[],"width":[],"height":[]},
                "bike":{"length":[],"width":[],"height":[]},
                "rider":{"length":[],"width":[],"height":[]},
                "car":{"length":[],"width":[],"height":[]},
                "truck": {"length": [], "width": [], "height": []},
                "bus": {"length": [], "width": [], "height": []},
                "sign": {"length": [], "width": [], "height": []},
                "Special":{"length":[],"width":[],"height":[]},
                 }
    for filename in data_list:
        full_label_path = os.path.join(data_dir,filename)
        with open(full_label_path, 'r', encoding="utf-8") as f:
            label_dict = json.load(f)['result']['data']
            num_of_objects = len(label_dict)
            # print(num_of_objects)

            for obj_id in range(num_of_objects):
                content = label_dict[obj_id]
                name = content['label']
                length = content['3Dsize']['height']
                width = content['3Dsize']['width']
                height = content['3Dsize']['deep']
                if name == "person":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                    if height > 1.8:
                        print(full_label_path)
                elif name == "pushing":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "bike":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "rider":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "car":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "truck":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "bus":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "sign":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)
                elif name == "Special":
                    data_dict[name]['length'].append(length)
                    data_dict[name]['width'].append(width)
                    data_dict[name]['height'].append(height)

    return data_dict


if __name__=="__main__":
    label_dir = './dataset_16568_dali/label/json/'
    label_list = os.listdir(label_dir)
    data_dict = get_distance(label_dir,label_list)

    for k1,v1 in data_dict.items():
        name = k1
        dims = v1
        # print(name,dims)
        for k2,v2 in data_dict[name].items():
            size = k2
            elements = v2
            # print(name,size,elements)
            distance_scatter_plot(name,size,elements)
            fit_normal_distribution(name,size,elements)