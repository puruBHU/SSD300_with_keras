#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:48:45 2019

@author: Purnendu Mishra
"""

import tensorflow as tf
import torch

import cv2
from pathlib import Path

from ssd300_model import SSD300
from skimage.io import imread
import numpy as np
from utility import *

import collections

from SSD_generate_anchors import generate_ssd_priors
import matplotlib.pyplot as plt
#from Detector import Detect
from detection import Detect

#%%****************************************************************************
target_size = (300,300)

mean = np.array([114.02898, 107.86698,  99.73119], dtype=np.float32)
std  = np.array( [69.89365, 69.07726, 72.30074], dtype=np.float32)

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

#%%****************************************************************************
SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

Spec = collections.namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 
                                       'aspect_ratios'])

# the SSD orignal specs
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]

priors = generate_ssd_priors(specs).astype(np.float32)

#%%****************************************************************************
def preprocess_image(image_path, target_size=(300, 300), mean = None, std = None):
    image = imread(image_path)
    img = cv2.resize(image, (300, 300), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img,axis = 0)
    img = np.float32(img)
#    img -= mean
    img /= 255.0
    return image, img

def TransformCoordinates(boxes = None, image_size = (300, 300)):
    xmin = np.maximum(0, boxes[:,0]) * image_size[1]
    ymin = np.maximum(0, boxes[:,1]) * image_size[0]
    
    xmax = np.maximum(0, boxes[:,2]) * image_size[1]
    ymax = np.maximum(0, boxes[:,3]) * image_size[0]
    
    return (xmin.astype(np.int16), ymin.astype(np.int16), xmax.astype(np.int16), ymax.astype(np.int16))


#%%****************************************************************************

image_path  =  Path.cwd()/'test_images'/'000005.jpg'

image, input_img = preprocess_image(image_path = image_path,
                                    mean = mean, 
                                    std = std)

h, w,c = image.shape
#%%****************************************************************************


   
model = SSD300(input_shape = (300, 300, 3), 
               anchors     = [4, 6,6,6,4,4], 
               num_classes = 21)

model.load_weights('checkpoints/EXP_01_SSD300_VOC_BS-16_EP-100_ChkPt_0.0523.hdf5', by_name= True)

prediction  = model.predict(input_img) 

#%%****************************************************************************    
loc_data    = prediction[:,:,:4]
conf_data   = prediction[:,:,4:]

loc_data   = torch.from_numpy(loc_data).float()
conf_data  = torch.from_numpy(conf_data).float() 
priors     = torch.from_numpy(priors).float() 

Detector   =  Detect(num_classes=21, bkg_label  = 0, top_k=200, conf_thresh=0.01, nms_thresh=0.2)
detections =  Detector.forward(loc_data = loc_data, conf_data=conf_data, prior_data=priors)

#print(np.unique(detections[:,:,:,0]))

labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

coordinates = []

for i in range(detections.shape[1]):
    j = 0
    while detections[0,i,j,0] >= 0.60:
#        print('Hello')
        score = [detections[0,i,j,0]]
        label_name = [str(labelmap[i-1])]
        pt         = (detections[0,i,j,1:])
        coords     = [pt[0], pt[1], pt[2], pt[3]]
        coords.append(label_name)
        coords.append(score)
        coordinates.append(coords)
        
        j += 1
        
        
    for coord in coordinates:
        xt, yt, xb, yb, name, score = coord[:]
        xmin = int(xt * w)
        ymin = int(yt * h)
        xmax = int(xb * w)
        ymax = int(yb * h)
            
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        
#        cv2.putText(img = image, 
#                    text = '{} ({:4f})'.format(name.upper(), score), 
#                    org = (xmin + 2, ymin + 15), 
#                    fontFace = font,
#                    fontScale = 1,
#                    color = (0,255,0))
        
plt.imshow(image)
plt.show()
        
