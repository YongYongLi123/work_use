import os
import json
import glob
import time
import shutil
import pprint
import struct
import numpy as np
from tqdm import tqdm
from pypcd import PointCloud


def read_examples_list(path):
    with open(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


# 1.第一批衡阳3260帧数据坐标系由(前+x左+y)转换到(右+x前+y),点云数据由二进制转化到ascii
is_transform = False
if is_transform:
    src_json_path = "./dataset_3260_hengyang/label_src/json/"
    src_pcd_path = "./dataset_3260_hengyang/pcd_src/"
    rotation_matrix = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

    json_list = os.listdir(src_json_path)
    json_list.sort()
    for filename in tqdm(json_list):
        src_json_filename = src_json_path + filename
        dst_json_filename = src_json_filename.replace("/label_src", "/label")
        src_pcd_filename  = src_pcd_path + filename.replace(".json",".pcd")
        dst_pcd_filename  = src_pcd_filename.replace("/pcd_src","/pcd")

        # json处理
        with open(src_json_filename, 'r') as fid:
            json_data = json.load(fid)
        new_json_data = json_data.copy()
        results = json_data['result']['data']
        new_results = list()
        for res in results:
            new_res = res.copy()
            center3D = np.array([res['3Dcenter']['x'],res['3Dcenter']['y'],res['3Dcenter']['z']])
            new_center3D = np.dot(rotation_matrix,center3D)
            new_res['3Dcenter']['x'] = new_center3D[0]
            new_res['3Dcenter']['y'] = new_center3D[1]
            new_res['3Dcenter']['z'] = new_center3D[2]

            alpha = res['3Dsize']['alpha'] + np.pi / 2
            if alpha > np.pi:
                heading = alpha - 2*np.pi
            new_res['3Dsize']['alpha'] = round(alpha,6)
            new_res['3Dsize']['rz'] = round(alpha,6)

            new_points = list()
            for pt in res['points']:
                new_pt = pt.copy()
                point = np.array([pt['x'],pt['y'],pt['z']])
                new_point = np.dot(rotation_matrix,point)
                new_pt['x'] = new_point[0]
                new_pt['y'] = new_point[1]
                new_pt['z'] = new_point[2]
                new_points.append(new_pt)
            new_res["points"] = new_points
            new_results.append(new_res)

        new_json_data['result']['data'] = new_results
        data = json.dumps(new_json_data, indent=1)
        with open(dst_json_filename, 'w', newline='\n') as k:
            k.write(data)

        # pcd处理
        cloud = PointCloud.from_path(src_pcd_filename)
        # pprint.pprint(cloud.get_metadata())
        new = cloud.pc_data.copy()
        acc = np.array([list(new) for new in new])
        f = open(dst_pcd_filename, 'w')
        lines = list(range(11))
        lines[0] = "# .PCD v0.7 - Point Cloud Data file format"
        lines[1] = "VERSION 0.7"
        lines[2] = "FIELDS x y z intensity"
        lines[3] = "SIZE 4 4 4 4"
        lines[4] = "TYPE F F F F"
        lines[5] = "COUNT 1 1 1 1"
        lines[6] = "WIDTH {}".format(acc.shape[0])
        lines[7] = "HEIGHT 1"
        lines[8] = "VIEWPOINT 0 0 0 1 0 0 0"
        lines[9] = "POINTS {}".format(acc.shape[0])
        lines[10] = "DATA ascii"
        for i in range(len(lines)):
            f.write('{}'.format(lines[i]))
            f.write('\n')
        for index in range(acc.shape[0]):
            data = np.dot(rotation_matrix,acc[index][:3])
            f.write('{:.6f} {:.6f} {:.6f} {:}'.format(float(data[0]),float(data[1]),float(data[2]),int(acc[index][3])))
            f.write('\n')
        f.close()


# 2.提取json文件中的3D框信息并且保存为txt
is_json2txt = False
is_Statistics_name = False
if is_json2txt:
    name_dict = dict()
    data_dir = "./dataset_x_dali_0915/label/"
    src_json_path = data_dir + "json/"
    dst_txt_path  = data_dir + "txt/"
    json_list = os.listdir(src_json_path)
    # json_list.sort(key=lambda x: int(x.split(".")[0]), reverse=False)
    json_list.sort()
    for filename in tqdm(json_list):
        src_json_filename = src_json_path + filename
        dst_txt_filename = src_json_filename.replace("/json", "/txt").replace(".json",".txt")
        # json处理
        fd = open(dst_txt_filename, 'w')
        with open(src_json_filename, 'r') as fid:
            json_data = json.load(fid)
        results = json_data['result']['data']
        for res in results:
            xc = res['3Dcenter']['x']
            yc = res['3Dcenter']['y']
            zc = res['3Dcenter']['z']
            dx = res['3Dsize']['height'] #length
            dy = res['3Dsize']['width']  #width
            dz = res['3Dsize']['deep']   #height
            heading = res['3Dsize']['alpha'] #偏航角(以x轴为初始位置，逆时针为正)
            name = res['label']
            # if name == "bus":
            #     print(filename)
            #     src_img = "./dataset_2006_dali/img/" + filename.replace(".json",".jpg")
            #     dst_img = "./bus_mini/img/" + filename.replace(".json",".jpg")
            #     src_json = data_dir + "json/" + filename
            #     dst_json = "./bus_mini/json/" + filename
            #     src_pcd = "./dataset_2006_dali/pcd/" + filename.replace(".json", ".pcd")
            #     dst_pcd = "./bus_mini/pcd/" + filename.replace(".json", ".pcd")
            #     shutil.copy(src_img, dst_img)
            #     shutil.copy(src_json,dst_json)
            #     shutil.copy(src_pcd,dst_pcd)
            if is_Statistics_name and name not in list(name_dict.keys()):
                name_dict[name] = 1
            elif is_Statistics_name:
                name_dict[name] += 1
            # if name in ['Cyclist', 'Tricyclist', 'Motorcyclist', 'Barrowlist']:
            #     name = 'Cyclist'
            #     fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            #     fd.write('\n')
            # elif name in ["Truck", "Van"]:
            #     name = "Truck"
            #     fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            #     fd.write('\n')
            # elif name == "Trafficcone":
            #     name = 'ignore'
            #     fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            #     fd.write('\n')
            # else: #["Car", "Bus", "Pedestrian"]
            #     fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            #     fd.write('\n')
            if name == "patrol cars":
                name = "patrol_cars"
            if name == "Sightseeing car":
                name = "Sightseeing"
            fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            fd.write('\n')
        fd.close()
    print(name_dict)


# 3.去掉pcd文件中的ring timestamp字段
is_Split_Special_Fields = False
if is_Split_Special_Fields:
    src_cloud_path = "./dataset_x_dali_0915/3d_url/"
    cloud_list = os.listdir(src_cloud_path)
    cloud_list.sort()

    dst_cloud_path = "./dataset_x_dali_0915/pcd/"
    dst_list = os.listdir(dst_cloud_path)
    dst_list.sort()

    for filename in tqdm(cloud_list):
        if filename in dst_list:
            continue
        src_cloud_filename = src_cloud_path + filename
        dst_cloud_filename = src_cloud_filename.replace("/3d_url","/pcd")

        # pcd处理
        f = open(dst_cloud_filename, 'w')
        lines = [line.rstrip() for line in open(src_cloud_filename)]
        lines[2] = "FIELDS x y z intensity"
        lines[3] = "SIZE 4 4 4 4"
        lines[4] = "TYPE F F F F"
        lines[5] = "COUNT 1 1 1 1"
        for i in range(len(lines)):
            if i < 11:
                f.write('{}'.format(lines[i]))
                f.write('\n')
            else:
                data = lines[i].split(' ')
                f.write('{:6f} {:.6f} {:.6f} {:}'.format(float(data[0]), float(data[1]), float(data[2]), int(data[3])))
                f.write('\n')
        f.close()


# 4.从数据集提取验证集
is_get_valset = False
if is_get_valset:
    dirname_read = '../dstData/npy/'
    dirname_write='../dstData/val_npy/'
    eval_txt = '../dstData/ImageSets/val.txt'
    files = os.listdir(dirname_read)
    files.sort()
    examples_list = read_examples_list(eval_txt)
    for file in examples_list:
        read_path = os.path.join(dirname_read, file) + '.npy'
        write_path = os.path.join(dirname_write, file) + '.npy'
        shutil.copy(read_path, write_path)


# 5.提取可视化验证集
is_get_show_val = False
if is_get_show_val:
    src_img_path = "./dataset_3260_hengyang/img/"
    src_pcd_path = "./dataset_3260_hengyang/pcd/"
    dst_img_path = "./dataset_3260_hengyang/show_val_img/"
    dst_pcd_path = "./dataset_3260_hengyang/show_val_pcd/"
    eval_txt = '../dstData/ImageSets/val.txt'
    files = os.listdir(src_img_path)
    files.sort()
    examples_list = read_examples_list(eval_txt)
    for file in examples_list:
        src_img_filename = src_img_path + file + ".jpg"
        src_pcd_filename = src_pcd_path + file + ".pcd"
        dst_img_filename = src_img_filename.replace('/img','/show_val_img')
        dst_pcd_filename = src_pcd_filename.replace('/pcd','/show_val_pcd')
        shutil.copy(src_img_filename,dst_img_filename)
        shutil.copy(src_pcd_filename,dst_pcd_filename)


# 6.解析大理测试rosbag数据
is_resolving_rosbag = False
if is_resolving_rosbag:
    data_root_path = "./dataset_dali_0315_test/"
    src_pcd_path = data_root_path + "src_pcd/"
    dst_pcd_path = src_pcd_path.replace("/src_pcd", "/pcd")
    rotation_matrix = np.array([[0.038611630083451684, 0.9914411047328892, -0.12471438516998043],
                                [-0.03813737156531759, 0.12617864518543498, 0.9912741751852802],
                                [0.9985262554947605, -0.03351843291676516, 0.042682920968657945]])
    loation_matrix = np.array([0.07611885113863269, 0.05234157192362775, 2.867007993608227])
    
    src_pcd_list = os.listdir(src_pcd_path)
    src_pcd_list.sort()

    for filename in tqdm(src_pcd_list):
        src_pcd_filename = src_pcd_path + filename
        dst_pcd_filenmae = src_pcd_filename.replace("/src_pcd","/pcd")

        cloud = PointCloud.from_path(src_pcd_filename)
        # pprint.pprint(cloud.get_metadata())
        new = cloud.pc_data.copy()
        acc = np.array([list(new) for new in new])
        f = open(dst_pcd_filenmae, 'w')
        lines = list(range(11))
        lines[0] = "# .PCD v0.7 - Point Cloud Data file format"
        lines[1] = "VERSION 0.7"
        lines[2] = "FIELDS x y z intensity"
        lines[3] = "SIZE 4 4 4 4"
        lines[4] = "TYPE F F F F"
        lines[5] = "COUNT 1 1 1 1"
        lines[6] = "WIDTH {}".format(acc.shape[0])
        lines[7] = "HEIGHT 1"
        lines[8] = "VIEWPOINT 0 0 0 1 0 0 0"
        lines[9] = "POINTS {}".format(acc.shape[0])
        lines[10] = "DATA ascii"
        for i in range(len(lines)):
            f.write('{}'.format(lines[i]))
            f.write('\n')
        for index in range(acc.shape[0]):
            data = np.dot(rotation_matrix,acc[index][:3]) + loation_matrix
            f.write('{:.6f} {:.6f} {:.6f} {:}'.format(float(data[0]),float(data[1]),float(data[2]),int(acc[index][7])))
            f.write('\n')
        f.close()


# 7.pcd转bin
is_pcd2bin = False
if is_pcd2bin:
    pcd_dir = "./bus_guiying/train_dl_3d_road_test_32_20230612_gy/pcd/"
    pcd_list = os.listdir(pcd_dir)
    pcd_list.sort()
    for filename in tqdm(pcd_list):
        pcd_filename = pcd_dir + filename
        bin_filename = pcd_filename.replace("/pcd","/bin").replace(".pcd",".bin")

        pt_lines = [pt_line.rstrip() for pt_line in open(pcd_filename)]
        points = np.full((len(pt_lines) - 11, 4), 255, dtype=np.float32)
        for rows in range(len(pt_lines)):
            if rows > 10:
                points[rows - 11, :4] = [float(pt_data) for pt_data in pt_lines[rows].split(' ')]

        points[:, 3] = points[:, 3] / 255
        points[:,:4].tofile(bin_filename)


# 8.过滤训练数据集中的无效点云帧(针对1月份验收数据)
is_filter = False
if is_filter:
    use_dir = "./filter/"
    json_dir = "./dataset_10267_dali/json/"
    pcd_dir = json_dir.replace("json/","pcd/")
    img_dir = json_dir.replace("json/","img/")
    json_list = os.listdir(json_dir)
    img_list = os.listdir(img_dir)
    pcd_list = os.listdir(pcd_dir)
    for filename in tqdm(json_list):
        pcd_filename = filename.replace(".json", ".pcd")
        img_filename = filename.replace(".json", ".jpg")
        if pcd_filename in pcd_list:
            src_pcd_path = pcd_dir + pcd_filename
            dst_pcd_path = use_dir + 'pcd/' + pcd_filename
            shutil.move(src_pcd_path,dst_pcd_path)
        if img_filename in img_list:
            src_img_path = img_dir + img_filename
            dst_img_path = use_dir + 'img/' + img_filename
            shutil.move(src_img_path, dst_img_path)


# 9.依照新标注规范提取json内容
is_json2txt = False
if is_json2txt:
    name_dict = {"person": 0, "pushing": 0, "bike":0, "rider":0,
                 "car":0, "truck":0, "bus":0, "sign":0, "Special":0,
                 "shed":0, "shed-non-motor":0,
                }
    data_dir = "./dataset_x_dali_0915/label/"
    src_json_path = data_dir + "json/"
    dst_txt_path  = data_dir + "txt/"
    json_list = os.listdir(src_json_path)
    # json_list.sort(key=lambda x: int(x.split(".")[0]), reverse=False)
    json_list.sort()
    for filename in tqdm(json_list):
        src_json_filename = src_json_path + filename
        dst_txt_filename = src_json_filename.replace("/json", "/txt").replace(".json",".txt")
        # json处理
        fd = open(dst_txt_filename, 'w')
        with open(src_json_filename, 'r') as fid:
            json_data = json.load(fid)
        results = json_data['result']['data']
        for res in results:
            xc = res['3Dcenter']['x']
            yc = res['3Dcenter']['y']
            zc = res['3Dcenter']['z']
            dx = res['3Dsize']['height'] #length
            dy = res['3Dsize']['width']  #width
            dz = res['3Dsize']['deep']   #height
            heading = res['3Dsize']['alpha'] #偏航角(以x轴为初始位置，逆时针为正)
            name = res['label']
            subname = res['sublabel']

            if name == "person":
                name_dict[name] += 1
            elif name == "pushing":
                name_dict[name] += 1
            elif name == "bike" and subname in ["bicycle","motorcycle","tricycle"]:
                name_dict[name] += 1
            elif name == "rider" and subname in ["bicycle","motorcycle","tricycle"]:
                name_dict[name] += 1
            elif name == "car":
                if subname == "shed" : #有棚机动车
                    name_dict[name] += 1
                    name_dict["shed"] += 1
                elif subname in ["mini","car","suv","minibus"]:
                    name_dict[name] += 1
                elif subname == "shed-non-motor": #有棚非机动车
                    name = "rider"
                    name_dict["shed-non-motor"] += 1
                    name_dict["rider"] += 1
            elif name == "truck":
                name_dict[name] += 1
            elif name == "bus":
                name_dict[name] += 1
            elif name == "sign" and subname == "conebucket":
                name_dict[name] += 1
            elif name == "Special":
                name_dict[name] += 1

            else:
                continue
            fd.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, xc, yc, zc, dx, dy, dz, heading))
            fd.write('\n')
        fd.close()
    print(name_dict)

# 10.生成索引值ImageSets
is_generate_data_index = False
if is_generate_data_index:
    txt_path = "../bus_mini/json/"
    txt_list = os.listdir(txt_path)
    train_txt_index = "../bus_mini/ImageSets/train.txt"
    print(train_txt_index)
    ftrain = open(train_txt_index, 'w')
    for inputfile in txt_list:
        filename = os.path.splitext(inputfile)[0]
        ftrain.write(filename)
        ftrain.write("\n")
    ftrain.close()

# 11.截取y方向65米之后的点云
is_intercept_point_cloud = False
if is_intercept_point_cloud:
    src_pcd = "./dataset_503_dali/pcd/"
    dst_pcd = "./dataset_503_dali/pcd_65/"
    src_label = "./dataset_503_dali/label/txt/"
    dst_label = "./dataset_503_dali/label_65/"

    src_pcd_list = os.listdir(src_pcd)
    # for filename in tqdm(src_pcd_list):
    #     src_pcd_filename = src_pcd + filename
    #     dst_pcd_filename = dst_pcd + filename
    #
    #     f = open(dst_pcd_filename, 'w')
    #     pt_lines = [pt_line.rstrip() for pt_line in open(src_pcd_filename)]
    #     points = np.full((len(pt_lines) - 11, 4), 255, dtype=np.float32)
    #     for rows in range(len(pt_lines)):
    #         if rows > 10:
    #             points[rows - 11, :4] = [float(pt_data) for pt_data in pt_lines[rows].split(' ')]
    #     mask_indices = points[:,1] > 65
    #     mask_points = points[mask_indices]
    #
    #     pcd_heads = pt_lines[:11]
    #     pcd_heads[6] = "WIDTH {}".format(mask_points.shape[0])
    #     pcd_heads[9] = "POINTS {}".format(mask_points.shape[0])
    #     for i in range(len(pcd_heads)):
    #         f.write('{}'.format(pcd_heads[i]))
    #         f.write('\n')
    #     for j in range(mask_points.shape[0]):
    #             f.write('{:6f} {:.6f} {:.6f} {:}'.format(mask_points[j][0], mask_points[j][1],
    #                                                      mask_points[j][2], int(mask_points[j][3])))
    #             f.write('\n')
    #     f.close()

    src_label_list = os.listdir(src_label)
    for filename_1 in tqdm(src_label_list):
        src_label_filename = src_label + filename_1
        dst_label_filename = dst_label + filename_1

        fid = open(dst_label_filename,"w")
        label_lines = [label_line.rstrip() for label_line in open(src_label_filename)]
        data_lines = [label.split(" ") for label in label_lines]
        for data in data_lines:
            if float(data[2]) > 65:
                fid.write(
                    '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(data[0],
                                                                                 float(data[1]), float(data[2]), float(data[3]),
                                                                                 float(data[4]), float(data[5]), float(data[6]),
                                                                                 float(data[7])))
                fid.write('\n')
        fid.close()


# 12.过滤训练数据集中的无效点云帧(针对1月份验收数据)
is_filter = True
if is_filter:
    use_dir = "./dataset_x_dali_0915/use/"
    json_dir = "./dataset_x_dali_0915/json/"
    pcd_dir = json_dir.replace("json/","pcd/")
    img_dir = json_dir.replace("json/","img/")
    json_list = os.listdir(json_dir)
    img_list = os.listdir(img_dir)
    pcd_list = os.listdir(pcd_dir)

    for filename in json_list:
        pcd_filename = filename.replace(".json", ".pcd")
        img_filename = filename.replace(".json", ".jpg")
        if pcd_filename in pcd_list:
            src_pcd_path = pcd_dir + pcd_filename
            dst_pcd_path = use_dir + 'pcd/' + pcd_filename
            shutil.move(src_pcd_path,dst_pcd_path)
        if img_filename in img_list:
            src_img_path = img_dir + img_filename
            dst_img_path = use_dir + 'img/' + img_filename
            shutil.move(src_img_path,dst_img_path)


# 13.整理不合规数据，返修(v2.5标注规范)
is_data_rework = False
if is_data_rework:
    src_data_dir = "./dataset_16568_dali/"
    dst_data_dir = "/home/ly/Work/parkCode/rework_data_v4.5/"
    src_json_list = os.listdir(src_data_dir + "label/json/")
    for filename in src_json_list:
        full_src_json_filename = src_data_dir + "label/json/" + filename
        full_dst_json_filename = dst_data_dir + "json/" + filename

        full_src_pcd_filename = src_data_dir + "pcd/" + filename.replace(".json",".pcd")
        full_dst_pcd_filename = dst_data_dir + "pcd/" + filename.replace(".json",".pcd")

        full_src_img_filename = src_data_dir + "img/" + filename.replace(".json",".jpg")
        full_dst_img_filename = dst_data_dir + "img/" + filename.replace(".json",".jpg")

        with open(full_src_json_filename, 'r') as fid:
            json_data = json.load(fid)
        results = json_data['result']['data']
        for res in results:
            name = res['label']
            length = res["3Dsize"]["height"]
            # v2.5标注规范数据
            # if name == "Car" or name == "Van":
            #     if length < 3.6 or length > 5.5:
            #         shutil.copy(full_src_json_filename,full_dst_json_filename)
            #         shutil.copy(full_src_img_filename,full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename,full_dst_pcd_filename)
            # elif name == "Truck":
            #     if length < 4.0 or length > 15:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "Bus" or name == "Sightseeing car" or name == "Sightseeing":
            #     if length < 4.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "Pedestrain":
            #     if length > 1.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "Cyclist" or name == "Tricyclist" or name == "Barrowlist" or name == "Motorcyclist":
            #     if length < 1.0 or length > 4.0:
            #         shutil.copy(full_src_json_filename,full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # v4.5标注规范数据
            # if name == "car":
            #     if length < 2.0 or length > 5.5:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "truck":
            #     if length < 4.0 or length > 15.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "bus":
            #     if length < 4.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "person":
            #     if length > 1.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            # elif name == "bike" or name == "rider":
            #     if length < 1.0 or length > 4.0:
            #         shutil.copy(full_src_json_filename, full_dst_json_filename)
            #         shutil.copy(full_src_img_filename, full_dst_img_filename)
            #         shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)

            if name == "truck":
                if length < 4.0 or length > 15:
                    print(full_src_json_filename)
                    shutil.copy(full_src_json_filename, full_dst_json_filename)
                    shutil.copy(full_src_img_filename, full_dst_img_filename)
                    shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
            elif name == "bus":
                if length < 3.0:
                    print(full_src_json_filename)
                    shutil.copy(full_src_json_filename, full_dst_json_filename)
                    shutil.copy(full_src_img_filename, full_dst_img_filename)
                    shutil.copy(full_src_pcd_filename, full_dst_pcd_filename)
