#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:24:14 2019

@author: Purnendu Mishra

"""
import torch
from box_utils import nms
import numpy as np
from utility import non_maximum_supression, decode

class Detect(object):
    def __init__(self, 
                 num_classes   = 21, 
                 bkg_label   = None, 
                 conf_thresh = 0.6, 
                 nms_thresh  = 0.6, 
                 top_k       = 200,
                 variances   = [0.1, 0.2]):
        
        self.num_classes = num_classes
        self.bkg_label   = bkg_label
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.top_k       = top_k
        self.variances    = variances
    
    def forward(self, loc_data, conf_data, priors):
        
#         loc_data   = prediction[:,:,:4]
#         conf_data  = prediction[:,:,4:]
        
        num_priors = priors.shape[0]
        batch_size = loc_data.shape[0]
        
        output  = np.zeros(shape=(batch_size, self.num_classes, self.top_k, 5), dtype= np.float32)
        
        conf_preds = conf_data.swapaxes(2,1)
        
        for i in range(batch_size):
            decoded_boxes = decode(loc       = loc_data[i], 
                                   priors    = priors,
                                   variances = self.variances)
            
            conf_scores = conf_preds[i].copy()
            
            
            for cl in range(1, self.num_classes):
                c_mask = np.greater(conf_scores[cl], self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                scores = np.float32(scores)
                
                if scores.shape[0] == 0:
                    continue
                
                l_mask =  c_mask.reshape(-1,1).repeat(4, axis= -1)   
                boxes  =  decoded_boxes[l_mask].reshape(-1,4).astype(np.float32) 
#                 print(boxes.shape)
                
                boxes     = torch.from_numpy(boxes).float()
                scores    = torch.from_numpy(scores).float()
                
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                
#                ids, count = non_maximum_supression(boxes    = boxes,
#                                                    scores   = scores, 
#                                                    overlap  = self.nms_thresh,
#                                                    top_k    = self.top_k)
##                
#                 print(ids.shape)
#                 print(count)
                ids = np.int32(ids)
                count = np.int32(count)
                
                scores = scores[ids[:count]]
                scores = np.expand_dims(scores, axis=1)
                
                output[i, cl, :count] = np.concatenate((scores, 
                                                         boxes[ids[:count]]), axis=-1)
                
#         flt = output.ascontiguousarray().reshape(batch_size, -1, 5)
#         idx  = np.argsort(flt[:,:,0], axis=-1)
#         rank = np.argsort(idx, axis=-1)
        
#         flt[rank < self.top_k].ex
                
        return output
