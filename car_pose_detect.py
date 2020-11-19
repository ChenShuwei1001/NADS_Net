import cv2
from NADS_Net_Solver import NADS_Net_Solver, draw_person_pose
import os
if __name__ == '__main__':

    weights_path = 'work_space/model/nads_model.pth'
    nads_net = NADS_Net_Solver(weights_file = weights_path, training = False)
    car_dir= '/home/alanschen/dataset/car/'

    for img in os.listdir(car_dir):
        img = cv2.imread(car_dir+img)
        poses, _ = nads_net.detect(img)
        img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
        cv2.imshow('car pose detection', img)
        cv2.waitKey(0)