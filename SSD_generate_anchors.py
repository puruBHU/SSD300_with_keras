#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:41:26 2019

@author: Purnendu Mishra
code taken from :
    https://gist.github.com/qfgaohao/fde0e68ec5d2a893265977fc46042f67?source=post_page---------------------------
"""

import collections
import numpy as np
import itertools

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



def generate_ssd_priors(specs, image_size=300, clip=True):
    """Generate SSD Prior Boxes.
    
    Args:
        specs: Specs about the shapes of sizes of prior boxes. i.e.
            specs = [
                Spec(38, 8, SSDBoxSizes(30, 60), [2]),
                Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                Spec(3, 100, SSDBoxSizes(213, 264), [2]),
                Spec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.
    
    Returns:
        priors: a list of priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size (300x300).
    """
    boxes = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            boxes.append([
                x_center,
                y_center,
                w,
                h
            ])
            
            # big sized square box
            size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            boxes.append([
                x_center,
                y_center,
                w,
                h
            ])           
            
            # change h/w ratio of the small sized box
            # based on the SSD implementation, it only applies ratio to the smallest size.
            # it looks wierd.
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = np.sqrt(ratio)                  
                boxes.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio,
                ])
                boxes.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio,
                ])
            


    boxes = np.array(boxes)
    if clip:
        boxes = np.clip(boxes, 0.0, 1.0)
    return boxes


if __name__ == '__main__':
    boxes = generate_ssd_priors(specs = specs)
    print(boxes)