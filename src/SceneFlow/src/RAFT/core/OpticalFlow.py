from math import fabs
import sys
sys.path.append('core')

import argparse
import pdb #pdb.set_trace()
import os
import cv2
import glob
import torch.nn as nn
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imgfile):
    img = np.array(Image.open(imgfile)).astype(np.uint8)  #uint8是专门用于存储图像（包括RGB，灰度图像等），范围是从0–255

    print("1 ", type(img), img.shape)    #<class 'numpy.ndarray'> (480, 640, 3)
    img = torch.from_numpy(img).permute(2, 0, 1).float()  #permute函数改变图像的维度顺序
    
    print("2 ", type(img), img.shape)    #<class 'torch.Tensor'> torch.Size([3, 480, 640])
    return img[None].to(DEVICE)

def optical(image1, image2):
    # pdb.set_trace()
    ROOT_DIR = os.getcwd()
    
    ROOT_DIR = "/mnt/SceneFlow/src/RAFT"
    # print(ROOT_DIR)
    
    MODEL_DIR = os.path.join(ROOT_DIR, "models/raft-things.pth")
    PATH_DIR = os.path.join(ROOT_DIR, "TUM")
    
    dict = { 
    'model': MODEL_DIR, 
    'path': PATH_DIR, 
    'small': False, 
    'mixed_precision': False, 
    'alternate_corr': False
    }

    model = torch.nn.DataParallel(RAFT(dict))
    model.load_state_dict(torch.load(dict["model"]))
    
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()  #permute函数改变图像的维度顺序
        # print("1 ", type(image1), image1.shape)    #<class 'torch.Tensor'> torch.Size([3, 480, 640])
        image1 = image1[None].to(DEVICE)
        
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()  #permute函数改变图像的维度顺序
        # print("2 ", type(image2), image2.shape)    #<class 'torch.Tensor'> torch.Size([3, 480, 640])
        image2 = image2[None].to(DEVICE)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2) #变成torch.Size([1, 3, 480, 640])
        # print(type(image1))
        # pdb.set_trace()
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()  #flow_uv (np.ndarray): shape [H,W,2]
        flo_image = flow_viz.flow_to_image(flow_uv)
        
        # print(type(flo_uv[0][0][0]))
        # print(len(flo_uv))
        # print(flo_uv.shape)

        return flow_uv
        # return flo_image
    
    # with torch.no_grad():
    #     images = glob.glob(os.path.join('/mnt/SceneFlow/src/RAFT/TUM', '*.png')) + \
    #              glob.glob(os.path.join('/mnt/SceneFlow/src/RAFT/TUM', '*.jpg'))
        
    #     images = sorted(images)
    #     i= 0
    #     for imfile1, imfile2 in zip(images[:-1], images[1:]):
    #         image1 = load_image(imfile1)
    #         image2 = load_image(imfile2)
    #         padder = InputPadder(image1.shape)
    #         image1, image2 = padder.pad(image1, image2) #变成torch.Size([1, 3, 480, 640])
    #         # pdb.set_trace()
    #         flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    #         flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
    #         flo = flow_viz.flow_to_image(flo)
    #         cv2.imwrite("/mnt/SceneFlow/src/RAFT/TUM/results/%s.png" % i, flo)
    #         i += 1
    #         # save_image(flo)

    
if __name__ == "__main__":
    
    optical()
    
