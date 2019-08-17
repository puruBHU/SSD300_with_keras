#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:37:54 2018
Modified on Mon Jul 22

@author: Purnendu  Mishra
"""


import cv2
import numpy as np
import pandas as pd
from skimage import io,color

from keras import backend as K
from keras.utils import Sequence, to_categorical

#from keras.preprocessing import image
from pathlib import Path
from xml.etree import ElementTree as ET
from random import shuffle
from utility import match, point_form, center_form


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def load_image(path, color_space = None, target_size = None):
    """Loads an image as an numpy array
    
    Arguments:
        path: Path to image file
        target_size: Either, None (default to original size)
            or tuple of ints '(image height, image width)'
    """
    img = io.imread(path)
    
    if target_size:
        img = cv2.resize(img, target_size, interpolation = cv2.INTER_CUBIC)
        
                    
    return img 

class DataAugmentor(object):
    
    def __init__(self,
                rotation_range   = 0.,
                zoom_range       = 0.,
                horizontal_flip  = False,
                vertical_flip    = False,
                rescale          = None,
                data_format      = None,
                normalize = False,
                mean = None,
                std = None 
                ):
        
        if data_format is None:
            data_format = K.image_data_format()
            
        self.data_format = data_format
        
        if self.data_format == 'channels_last':
            self.row_axis = 0
            self.col_axis = 1
            self.channel_axis = 2
        
        self.rotation_range  = rotation_range
        self.zoom_range      = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.normalize       = normalize
        
        self.rescale = rescale
        self.mean = mean
        self.std = mean
        
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
            
        elif len(zoom_range) == 2:
            self.zoom_range =[zoom_range[0], zoom_range[1]]
            
        else:
            raise ValueError("""zoom_range should be a float or
                             a tuple or lis of two floats. 
                             'Receved args:'""", zoom_range)
        
    
    def random_transforms(self, samples, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            

            
            
        if len(samples) != 2:
            x = samples
            y = None
            
        else:
            x = samples[0]
            y = samples[1]
        
                    
        if self.rotation_range:
            theta = int(180 * np.random.uniform(-self.rotation_range,
                                                self.rotation_range))
            
            (h, w) = x.shape[:2]
            (cx, cy) = [w//2, h//2]
            
            M = cv2.getRotationMatrix2D((cx,cy), -theta, 1.0)
            x = cv2.warpAffine(x , M, (w,h))
            y = cv2.warpAffine(y,  M, (w,h))
            
            
        if self.horizontal_flip:    
            if np.random.random() < 0.5:
                x      = x[:,::-1,:] #flip x along x axis
    
                xc = y[:,0]
                y[:,0] = 1.0 - xc
                
                
        if self.vertical_flip:
            if np.random.random() < 0.5:
                print('Vertical flipped')
                x = x[::-1,:,:]
                # To flip the bounding box coordinate 
                # subtract center coordinates from 1.0
                # Assuming boxes are normalized and have form (xc, yc, h, w)
                yc = y[:,1]
                y[:,1] = 1 - yc
           
        
#        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
#            zx, zy = 1, 1
#        else:
#            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#            
#        x = image.random_zoom(x, (zx,zy), channel_axis=self.channel_axis)
#        y = image.random_zoom(y, (zx,zy), channel_axis=self.channel_axis)
        
        return (x,y)
    
    def flow_from_directory(self,
                            root        = None, 
                            data_folder   = None,
                            mode          = 'train',
                            target_size = (300, 300),
#                            color_space = None,
                            batch_size  = 8,
                            num_classes = 21,
                            shuffle     = False,
                            data_format = None,
                            seed        = None,
                            priors      = None):
        return Dataloader(
                    root,
                    self,
                    data_folder =  data_folder,
                    mode        = mode,
                    target_size = target_size,
#                    color_space = color_space,
                    batch_size  = batch_size,
                    num_classes = num_classes,
                    shuffle     = shuffle,
                    data_format = self.data_format,
                    seed        = seed,
                    priors      = priors
                    )
    
    def standardize(self,x):
        """Apply the normalization configuration to a batch of inputs.
            Arguments:
                x: batch of inputs to be normalized.
            Returns:
                The inputs, normalized.
        """       
        x = x.astype('float32')
        
        if self.rescale:
            x *= self.rescale
            
            
        if self.normalize:
            if self.mean is not None:
                x -= self.mean
#            else:
#                x -= np.mean(x, axis=self.channel_axis, keepdims=True)
                
                
            if self.std is not None:  
                x /= self.std
                
#            else:
#                x /= (np.std(x, axis=self.channel_axis, keepdims=True) + 1e-7)

        return x


class Dataloader(Sequence):
    ''' Data generator for keras to be used with model.fit_generator
    
    Args:
        root : (str) path to dataset folder
        data_folder: (slist) file containing the names of trainable data
        mode       : (str) train or validate
        batch_size: Integere, size of a batch
        shuffle: Boolean, whether to shuffle the data between epochs
        target_size = Either 'None' (default to original size) or
            tuple or int '(image height, image width)'
            
        
            
    
    Returns:
        tuple: Batch_x, Batch_y 
        shape  Batch_x: batch_size, image_height, image_width, 3
        shape  Batch_y: batch_size, no.of  priors, 4 + no.of classes      
    '''
    
    def __init__(self,
                 root        = None,
                 image_data_generator=None,
                 data_folder  = ['VOC2007', 'VOC2012'],
                 mode         = 'train',
                 batch_size   = None, 
                 shuffle      = True,
                 target_size  = None,
#                 color_space  = None, 
                 data_format  = 'channel_last',
                 num_classes  = 21,
                 seed         = None,
                 priors       = None):
        
#         super(Dataloader, self).__init__(self)
        
        if data_format is None:
            data_format = K.image_data_format()
        
        self.root_path           = root 
        self.image_data_generator =  image_data_generator
        self.batch_size         = batch_size
        self.shuffle            = shuffle
        self.num_classes        = num_classes
        self.target_size        = target_size
#        self.color_space        = color_space
        self.data_format        = data_format
        self.seed               = seed
        self.priors             = priors
        
        self.files = []
        
        if len(data_folder) == 0:
            raise ValueError('No path provide, please provide name of VOC dataste, for example ["VOC2007"] ')

#%%        
        for folder in data_folder:
            path    = self.root_path/folder
            
            if mode == 'train':
                data_file     = path/'ImageSets'/'Main'/'train.txt'
            elif mode == 'val':
                data_file     = path/'ImageSets'/'Main'/'val.txt'     
            elif mode == 'test':
                data_file     = path/'ImageSets'/'Main'/'test.txt'
        
            with open(data_file, 'r') as f:
                file_names = f.read().split()
                
            for t in file_names:
                temp = (folder, t)
                self.files.append(temp)
                
#%%            
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
            
        elif isinstance(target_size, tuple):
            self.target_size = target_size
            
        else:
            raise ValueError('Expected target_size to be either a int or a tuple')
        
        if data_format == 'channels_last':
            self.row_axis        = 1
            self.col_axis        = 2
            self.channel_axis    = 3
            self.image_shape     = self.target_size + (3,)
            
         
        self.on_epoch_end()

#%%        
    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))
    
#%%    
    def __getitem__(self, idx):

        # total number of samples in the dataset
        n = len(self.files)
        
        if n > idx * self.batch_size:
            current_batch_size = self.batch_size
        else:
            current_batch_size = n - self.batch_size
        
        
        file_names = self.files[idx * current_batch_size : (idx + 1) * current_batch_size]
#         print batch_x.shape
        batch_x = []
        batch_y = []
        
        num_priors = self.priors.shape[0]
        
        for m, files in enumerate(file_names):
            
            labels           = np.zeros(shape = (num_priors, self.num_classes + 4), dtype = np.float32)
                       
            image_path       = self.root_path/files[0]/'JPEGImages'/files[1]
            annotation_path  = self.root_path/files[0]/'Annotations'/files[1]
                        
            image_file        = image_path.with_suffix('.jpg')
            annotation_file   = annotation_path.with_suffix('.xml')
            
            # Read the image
            image = load_image(image_file, target_size = self.target_size)
            image = np.array(image, dtype = np.float32)
            
            
            # Get the ground truth
            self.ReadVOCAnnotations(annotation_file = annotation_file)
            
            ground_truth = np.array(self.TransformBNDBoxes(), dtype=np.float32)
            
            # Data Augmentation
            image, ground_truth[:,1:] = self.image_data_generator.random_transforms((image, ground_truth[:,1:]))
            
            # Data normalization
            image     = self.image_data_generator.standardize(image)
            
            bndbox_loc = ground_truth[:,1:]
            class_ids  = ground_truth[:,0]
            
            loc, class_id  = match(truths = point_form(bndbox_loc), # Convert to from (xmin, ymin, xmax, ymax) 
                                   labels = class_ids,
                                   priors = self.priors, 
                                   variance= [0.1, 0.2], 
                                   threshold = 0.5)
            
            class_id  = to_categorical(class_id, num_classes=self.num_classes)
            
            labels[:,:4] = loc
            labels[:,4:] = class_id
            
          
            
            batch_x.append(image)
            batch_y.append(labels)   
            
        batch_x = np.array(batch_x, dtype = np.float32)
        batch_y = np.array(batch_y, dtype = np.float32)
        
        
        return batch_x, batch_y
            
    
    def on_epoch_end(self):
        'Shuffle the at the end of every epoch'
       
        if self.shuffle == True:
            shuffle(self.files) 
            
    def ReadVOCAnnotations(self, annotation_file):
        self.root = ET.parse(annotation_file).getroot()
    
    def GetBNDBoxes(self):
        data  = []   #empty list to hold bounding box coordinates
        for elements in self.root.findall('./object/bndbox'):
            xmin = int(float(elements.find('xmin').text))
            ymin = int(float(elements.find('ymin').text))
            xmax = int(float(elements.find('xmax').text))
            ymax = int(float(elements.find('ymax').text))
            
            data.append([xmin, ymin, xmax, ymax])
            
        return data
    
    def GetImageSize(self):
        for f in self.root.findall('./size'):
            height = int(f.find('height').text)
            width  = int(f.find('width').text)
            depth  = int(f.find('depth').text)
        
        return {'height':height,'width':width, 'depth':depth}
    
    def GetObjectClass(self):
        objects = []
        for elements in self.root.findall('./object'):
            objects.append(elements.find('name').text)
            
        return objects
    
    def TransformBNDBoxes(self):
        data = []
        boxes   = self.GetBNDBoxes()
        obj_class = self.GetObjectClass()
        dim     = self.GetImageSize()
        
        image_height = dim['height']
        image_width  = dim['width']
        
        for i, box in enumerate(boxes):
            ''' Normalizing bounding box dimensions with image spatial resolution'''
            xmin = box[0] / image_width
            ymin = box[1] / image_height
            
            xmax = box[2] / image_width 
            ymax = box[3] / image_height
            
            bndbox_height = ymax - ymin
            bndbox_width  = xmax - xmin
            
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
        
            
            # Since class_id = 0 is reserved for the background, 1 is added to index to genereate 
            # class_id for objects in the VOC dataset
            data.append([VOC_CLASSES.index(obj_class[i]) + 1.0, xc, yc, bndbox_width, bndbox_height])
            
        return np.array(data)
#%%  
if __name__ == '__main__':
#    root                = Path.home()/'data'/'VOCdevkit/VOC2007'
    root   = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'
#    voc_image_path      = root/'JPEGImages'
#    voc_annotation_path = root/'Annotations'
#    voc_trainval_path   = root/'ImageSets'/'Main'/'train.txt'
    
    tester = DataAugmentor()
    generator = tester.flow_from_directory(root        = root,
                                           data_folder = ['VOC2007','VOC2012'],
                                           target_size = 300,
                                           batch_size  = 1,
                                           shuffle = True)
    
    sample = generator[0]
    print(sample[0].shape)
    print(sample[1])
    
    