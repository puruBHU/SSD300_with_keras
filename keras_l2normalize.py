#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:26:29 2019

@author: Purnendu Mishra
"""

from keras import backend as K
from keras.engine.topology import InputSpec
from keras.initializers import Constant
from keras.layers import Layer
import numpy as np


class L2Norm(Layer):
    
    def __init__(self, scale = 20, axis = -1, **kwargs):
        
        self.channel_axis = axis
        self.scale        = scale
        
        super(L2Norm, self).__init__()
        
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        
        # Self.scale will
        self.gamma = self.add_weight(name  = '{}'.format(self.name),
                                     shape = (input_shape[self.channel_axis],),
                                     initializer = Constant(self.scale),
                                     trainable   = True)
        
        super(L2Norm, self).build(input_shape)
     
    def call(self, x):
        return self.gamma * K.l2_normalize(x, axis = self.channel_axis) 
    
    def compute_output_shape(self, input_shape):
        return input_shape
