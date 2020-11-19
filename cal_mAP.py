
from NADS_Net_Solver import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
import cv2

import os
import json

DEBUG_INFO = 'debug breakpoint'

#todo: compare with other coder's code
if __name__ == '__main__':

    coco_dir = '/home/alanschen/dataset/coco2017/'
    coco_val_dir = coco_dir+'val2017/'
    json_file_path = coco_dir + 'detected_kp.json'
    weights_path = 'work_space/model/nads_model.pth'

    nads_net = NADS_Net_Solver(weights_file = weights_path, training = False)

    # groundtruth
    coco_val = COCO(os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json'))

    detected_kp = []
    i, val_len = 0, len(coco_val.imgToAnns.keys())
    for img_id in coco_val.imgToAnns.keys():
        img_name = coco_val.imgs[img_id]['file_name']
        img = cv2.imread(coco_val_dir + img_name)
        poses, scores = nads_net.detect(img)
        if 0 != poses.size:
            poses = np.delete(poses,[1], axis=1)
        # the detected keypoints, i.e. 'poses', its order is different from the groundtruth's order
        # more details are in 'entity.py: class JointType'
            poses[:,[i for i in range(17)],:] = poses[:,[0,14,13,16,15,4,1,5,2,6,3,10,7,11,8,12,9],:]
        poses = poses.tolist()
        for idx, pose in enumerate(poses):
            coordinate = [item for sublist in pose for item in sublist]
            detected_kp.append({'image_id':img_id, 'keypoints':coordinate, 'category_id':1, 'score':scores[idx]})
        print('{}/{}'.format(i, val_len))
        i+=1
    json.dump(detected_kp, open(json_file_path, 'w'))

    res_file = coco_dir + 'detected_kp.json'
    Dt = coco_val.loadRes(res_file)

    coco_eval = COCOeval(cocoGt=coco_val, cocoDt=Dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
