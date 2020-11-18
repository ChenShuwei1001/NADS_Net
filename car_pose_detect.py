import cv2
import argparse
from NADS_Net_Solver import NADS_Net_Solver, draw_person_pose

if __name__ == '__main__':

    weights_path = 'work_space/model/nads_model.pth'
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='test img path')
    arg = parser.parse_args()

    # load model
    nads_net = NADS_Net_Solver(weights_file = weights_path, training = False)

    # read image and inference
    img = cv2.imread(arg.img)
    poses, _ = nads_net.detect(img)

    # draw and save image
    img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)