
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
    weights_path = 'work_space/model/nads_model.pth'
    json_file_path = coco_dir + 'detected_kp.json'
    # json_file_path = coco_dir + 'detected_kp2.json'
    # coco gt
    coco_val = COCO(os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json'))
    nads_net = NADS_Net_Solver(weights_file = weights_path, training = False)

    # file_name = coco_val.loadImgs(425226)[0]['file_name']
    # file_path = coco_val_dir + file_name
    # img = cv2.imread(file_path)
    # poses, _ = nads_net.detect(img)
    # img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
    # cv2.imshow('car pose detection', img)
    # cv2.waitKey(0)

    detected_kp = []
    i, l =0, len(coco_val.imgToAnns.keys())
    for img_id in coco_val.imgToAnns.keys():
        img_path = coco_val.imgs[img_id]['file_name']
        img = cv2.imread(coco_val_dir+img_path)
        poses, scores = nads_net.detect(img)
        if 0 != poses.size:
            poses = np.delete(poses,[1], axis=1)
        poses = poses.tolist()
        for idx, pose in enumerate(poses):
            coordinate = [item for sublist in pose for item in sublist]
            detected_kp.append({'image_id':img_id, 'keypoints':coordinate, 'category_id':1, 'score':scores[idx]})
        i+=1
        print('{}/{}'.format(i, l))
        # if(10==i):
        #     break
    json.dump(detected_kp, open(json_file_path, 'w'))
    # res_file = coco_dir + 'detected_kp.json'
    # Dt = coco_val.loadRes(res_file)
    Dt = coco_val.loadRes(json_file_path)
    coco_eval = COCOeval(cocoGt=coco_val, cocoDt=Dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate(coco_eval.params)
    coco_eval.summarize()
    print(DEBUG_INFO)
