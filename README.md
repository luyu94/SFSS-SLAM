# SFSS-SLAM

Visual simultaneous location and mapping (SLAM) plays an important role in navigation and augmented reality (AR). Static environment is a common assumption for traditional vSLAM to work properly, which limits the wide usage of vSLAM in real world scenes. The dynamic objects can impair the accuracy of camera pose estimation and the quality of point cloud map.

In this paper we propose **SFSS-SLAM**, a novel motion removal visual system built on **ORB-SLAM2**. The algorithm combines mature semantic segmentation network Mask-RCNN with scene flow we present for pixel-level dynamic objects extraction in RGB-D image sequences. The scene flow model uses dense optical flow RAFT and depth data to estimate undefined moving objects. Besides, the motion removal is accomplished by updating moving probability of each keypoint and remove outliers from tracking. In order to verify the performance of our SLAM system, we evaluate on public [TUM RGB-D datasets](http://vision.in.tum.de/data/datasets/rgbd-dataset) and the results demonstrate that our method improves the tracking accuracy and work robustly in dynamic environments.

We provide examples to run proposed SLAM system in the TUM dataset and sorce code to obtain scene flow.


![overview](https://github.com/luyu94/SFSS-SLAM/blob/master/images/overview.jpg)



![scene](https://github.com/luyu94/SFSS-SLAM/blob/master/images/scene.png)

# Prerequisites
## ORB-SLAM2
The system needs ORB-SLAM2 prerequisites and it can be found at: https://github.com/raulmur/ORB_SLAM2.
We tested the project in Ubuntu 18.04 using OpenCV 3.4.5，Eigen3 3.2.10 and CUDA 10.1.
## Mask-RCNN
Semantic segmentation Mask-RCNN is found at https://github.com/matterport/Mask_RCNN.
Install `python==3.6.9  tensorflow-gpu==2.1.0  keras==2.8.0`
## RAFT
Optical Flow network RAFT is found at https://github.com/princeton-vl/RAFT
Install `torch==1.6.0+cu101`

# Building library

Clone the repository:
```
git clone https://github.com/luyu94/SFSS-SLAM.git
cd SFSS-SLAM
chmod +x build.sh
./build.sh
```

#  Examples

## TUM RGB-D Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Execute the following command.
```
./Examples/RGB-D/rgbd_tum_scene Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml PATH_OF_DATASET ASSOCIATION_FILE /mnt/SFSS-SLAM/a_output/output_fw3_xyz &> log
```
for examle:
```
./Examples/RGB-D/rgbd_tum_scene Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /mnt/rgbd_dataset_freiburg3_walking_xyz /mnt/SFSS-SLAM/Examples/RGB-D/associations/fr3_walking_xyz.txt /mnt/SFSS-SLAM/a_output/output_fw3_xyz
```

# Acknowledgements
Our code builds on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)



