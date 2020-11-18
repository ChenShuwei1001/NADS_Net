
from NADS_Net_Solver import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

import os
import argparse
import json

DEBUG_INFO = 'debug breakpoint'

def trainsform_multi_pose(poses):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='test img path')
    arg = parser.parse_args()

    coco_dir = '/home/alanschen/dataset/coco2017/'
    # coco gt
    coco_val = COCO(os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json'))

    weights_path = 'work_space/model/nads_model.pth'
    nads_net = NADS_Net_Solver(weights_file = weights_path, training = False)
    coco_val_dir = coco_dir+'val2017/'
    detected_kp = []
    i=0
    for img_id in coco_val.imgToAnns.keys():
        img_path = coco_val.imgs[img_id]['file_name']
        img = cv2.imread(coco_val_dir+img_path)
        poses, _ = nads_net.detect(img)
        poses = poses.tolist()
        for pose in poses:
            coordinate = [item for sublist in pose for item in sublist]
            detected_kp.append({'image_id':img_id, 'keypoints':coordinate, 'category_id':1})

        i+=1
        print(i)
        if(10==i):
            break

    json_file_path = coco_dir + 'detected_kp.json'
    json.dump(detected_kp, open(json_file_path, 'w'))

    coco_val.loadRes(json_file_path)
