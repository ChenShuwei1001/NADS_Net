# NADS-Net: A Nimble Architecture for Driver and Seat Belt Detection via Convolutional Neural Networks



This is an **approximate** implementation of [NADS-Net: A Nimble Architecture for Driver and Seat Belt Detection via Convolutional Neural Networks](https://arxiv.org/abs/1910.03695) with Pytorch. 

The origin code of NADS-Net is not public. Considering that the work of NADS-Net is based on [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008) , we **approximately** implement the NADS-Net based on an implementation of OpenPose:[an implementation of PyTorch OpenPose](https://github.com/tracer9/Pytorch0.4.1_Openpose).



## Attention!

NADS-Net is a network  with there detection head, i.e. the keypoints detection head(outputs Confidence Maps), the PAF detection head, and a seat belt segmentation head. Our application scenario is to estimate the pose of the drivers to detect and prevent possible dangerous driving behaviors. Status of seat belts matters a lot in the scenario. However, there is **EXTREMELY FEW** public datasets with seat belt labeling. Considering that, the influence of safety belt on driving pose estimation is not included in our application, i.e. we **DID NOT** implement the seat belt detection head! We only detect the pose of the driver(by only the keypoints detection head and the PAF detection head) to assess driving safety.



## Content

1. Configure environment
2. Download trained model
3. Test
4. Train your own model using your data



## Configure environment

**git clone** and **cd into** the project:

```
git clone https://github.com/ChenShuwei1001/NADS_Net.git
cd NADS_Net
```

### Download the trained model

the trained model is placed in:

1. [google driver]()
2. [one drive]()
3. [pan baidu](https://pan.baidu.com/s/1sRzhS3EGpwNYcXpBbcEZHg) and the password: **aauu**

after downloading the file **nads_model.pth**, copy it into **work_space/model/**



Dependencies for testing and training are installed in this section. The version of all software packages is only used by the author when programming, and does not represent the version that must be used.  **Anaconda** is used, so create and activate an anaconda environment before configuring by using command:

```
conda create -n nads_net python=3.7
conda activate nads_net
```

### Install COCO python API

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../../
```

### Install other python package dependencies

```
pip install -r requirements.txt
```

## Test

Execute the following command with the image file as arguments for estimating pose. The resulting image will be saved as `result.jpg`.

```
python pose_detect.py --img=data/author.jpg
```

Give the origin test image and the inferenced image as follows:

![origin image](https://github.com/ChenShuwei1001/NADS_Net/blob/master/data/author.jpg)

![inferenced_image](https://github.com/ChenShuwei1001/NADS_Net/blob/master/data/result.png)

## Train

This is a training procedure using COCO 2017 dataset.

### Download COCO 2017 dataset

If you already downloaded the dataset by yourself, please skip this procedure below and change coco_dir in `entity.py` to the dataset path that was already downloaded.

```
bash getData.sh
```

### Generate and save image masks

Mask images are created in order to filter out people regions who were not labeled with any keypoints so that the model will not be punished by true positive but unlabeled prediction. `vis` option can be used to visualize the mask generated from each image.

```
python gen_ignore_mask.py
```

### Train with COCO dataset

use the command below to start training.

```
python train.py
```

Inside each epoch, the recent weight parameters are saved as a weight file. The number of times that the parameters to be saved is controlled by `save_interval`in `entity.py` file.

More configuration about training are in the `entity.py` file.

### AP&mAP

The script  `cal_mAP.py` calculates the AP&mAP value of the model by invoking COCOAPI. One thing you should know is that, **the order of the keypoints used in training stages is different from the order in COCO Dataset, the original author of the referenced openpose implementation made a mapping between them.** You should remap the order of the predicted keypoints with the order of groundtruth keypoints.

