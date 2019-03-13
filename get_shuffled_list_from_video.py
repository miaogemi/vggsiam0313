# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"


import cv2
import os
import os.path
from region_to_bbox import region_to_bbox_normalized
from random import shuffle
import numpy as np

def prepare_shuffled_list(data_folder, output_filename, output_directory, num_video):
    """
        Input:
            data_folder: relative path of folder who contains all the vedio folders for training.
            tfrecord_name: nameof the tfrecord file
            output_directory: relative dir which will contain the tfrecord file

    """
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
        
    
        
    cur_dir = os.getcwd()
    data_folder = os.path.join(cur_dir, data_folder)
    #get a list of dirs in data_folder, in each of which contains a training vedio
    #找到datafolder文件当中的每一个dir，所有不是可执行文件的文件（也就是那些文件夹）进行排序，生成一个文件列表
    video_folder_list = sorted([dir for dir in os.listdir(data_folder) if not os.path.isfile(os.path.join(data_folder, dir))])[:num_video]
    vedio_index = 0
    output_list = []

    #loop through all the vedio folders
    for video_folder in video_folder_list:
        print("Reading images from " + video_folder);
        vedio_index += 1
        
        #videofolederlist是视频文件列表，filelist是每个视频文件中的文件列表（包含txt、jpg等），imglist是每个视频文件中的图片列表
        video_folder = os.path.join(data_folder, video_folder)
        #get a list of dirs in data_folder, in each of which contains a training vedio
        file_list = [dir for dir in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, dir))]
        img_list = sorted([file for file in file_list if file.endswith(".jpg")])
        
        gt_file_name = "groundtruth.txt"        
        assert os.path.exists(os.path.join(video_folder, gt_file_name))        
        gt_file = open(os.path.join(video_folder, gt_file_name), 'r')
        gts = gt_file.readlines()
        #check if num of ground truth equals the num of img files
        assert len(gts) == len(img_list)
        _examples = list(zip(img_list, gts))#zip产生元组对，list将元组转换为列表[('01.jpeg', [910,366,80,322])]
        
        #prepare examplar z模板图像
        z = os.path.join(video_folder, img_list[0])
        z_raw = z.replace('\t',r'\t').replace('\n', r'\n').replace('\000', r'\000')
        z_img = cv2.imread(z_raw)
        z_gt = gts[0].strip("\n").split(",")#zgt数据代表，模板图像在大图像中的位置宽，高、模板图像宽，高   
        assert len(z_gt) == 4
        #x, y in ground truth is the coordinate of the topleft corner, we need to convert to the center
        #cx / width, cy / height, w / width, h / height
        #模板图像z中心的x，y位置（占原图的比例，原图为1），z的宽和高（也是占原图的比例）
        z_pos_x, z_pos_y, z_target_w, z_target_h = region_to_bbox_normalized(z_gt, z_img.shape[1], z_img.shape[0])
      
               
        for _example in _examples[1: ]:
            assert len(_example) == 2
            gt = _example[1].strip("\n").split(",") 
            assert len(gt) == 4
            
            x = os.path.join(video_folder, _example[0])
            x_img = cv2.imread(x)
                                                
            x_pos_x, x_pos_y, x_target_w, x_target_h = region_to_bbox_normalized(gt, x_img.shape[1], x_img.shape[0]) 
            
            
            output_list.append(z + " " + x + " " + str(z_pos_x)+ " "+ str(z_pos_y)+ " "+  str(z_target_w)+ " "+ str(z_target_h)+ " "+  str(x_pos_x)+ " "+ str(x_pos_y)+ " "+  str(x_target_w)+ " "+  str(x_target_h))
            z = x
            z_pos_x, z_pos_y, z_target_w, z_target_h = x_pos_x, x_pos_y, x_target_w, x_target_h
    print("Finished loading all the training data. Start shuffling...")
    shuffle(output_list);
    with open(os.path.join(output_directory,output_filename) + ".txt","w+") as f:
        for output_file in output_list:
            f.write(output_file + "\n") 
    print("Done. tfrecords file saved at " + output_directory)
        

    
if __name__ == "__main__":
    prepare_shuffled_list("cfnet-validation", "shuffled_data_list", "tfrecords", num_video = 78)
