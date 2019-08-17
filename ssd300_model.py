#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:12:43 2019

@author: Purnendu Mishra
"""
from keras.models import Model
from keras.layers import Input, MaxPooling2D, ZeroPadding2D
from keras.layers import Reshape, Add, Conv2D, Concatenate, Lambda, Activation
from utility import conv_bn_relu
from keras import backend as K
import tensorflow as tf

from keras_l2normalize import L2Norm

def SSD300(input_shape = (None, None, 3), anchors = [4, 6,6,6,4,4], num_classes = 21):
    input_ = Input(shape = input_shape)
    
    # Block 01
    x =  conv_bn_relu(filters = 64, 
                      name = {'conv':'conv1_1', 'batch_norm':'bn1_1','activation':'relu_1_1'})(input_)
    x =  conv_bn_relu(filters = 64, 
                      name = {'conv':'conv1_2', 'batch_norm':'bn1_2','activation':'relu_1_2'})(x)
    x =  MaxPooling2D(pool_size=(2,2), name = 'pool1')(x)
    
    # Block 02
    x = conv_bn_relu(filters = 128, 
                     name = {'conv':'conv2_1', 'batch_norm':'bn2_1','activation':'relu_2_1'})(x)
   
    x = conv_bn_relu(filters = 128, 
                     name = {'conv':'conv2_2', 'batch_norm':'bn2_2','activation':'relu_2_2'})(x)
     
    x = MaxPooling2D(pool_size = (2,2), name='pool2')(x)
    
    # Block 3
    x = conv_bn_relu(filters = 256, 
                     name = {'conv':'conv3_1', 'batch_norm':'bn3_1','activation':'relu_3_1'})(x)
    
    x = conv_bn_relu(filters = 256, 
                     name = {'conv':'conv3_2', 'batch_norm':'bn3_2','activation':'relu_3_2'})(x)
    
    x = conv_bn_relu(filters = 256, 
                     name = {'conv':'conv3_3', 'batch_norm':'bn3_3','activation':'relu_3_3'})(x)
    
    # Apdding zeros to get layer shape as mentioned in the paper
    x = ZeroPadding2D(padding = (1,1), name = 'zero_01')(x)
    
    x = MaxPooling2D(pool_size = (2,2), name='pool3')(x)
    
    # Block 4
    x = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv4_1', 'batch_norm':'bn4_1','activation':'relu_4_1'})(x)
    
    x = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv4_2', 'batch_norm':'bn4_2','activation':'relu_4_2'})(x)
    
    conv4_3 = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv4_3', 'batch_norm':'bn4_3','activation':'relu_4_3'})(x)
    
    x = MaxPooling2D(pool_size = (2,2), name='pool4')(conv4_3)
    
    # Block 5
    x = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv5_1', 'batch_norm':'bn5_1','activation':'relu_5_1'})(x)
    
    x = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv5_2', 'batch_norm':'bn5_2','activation':'relu_5_2'})(x)
    
    x = conv_bn_relu(filters = 512, 
                     name = {'conv':'conv5_3', 'batch_norm':'bn5_3','activation':'relu_5_3'})(x)
    
    # Pool5 should be with pool_size = (3 x 3) and strides= (1,1)
    x = MaxPooling2D(pool_size = (3,3),strides = (1,1), padding='same', name = 'pool5')(x)
    
    #Auxilary Layer
    fc6 = conv_bn_relu(filters = 1024, 
                      name = {'conv':'conv_fc6', 'batch_norm':'bn_fc6','activation':'relu_fc6'})(x)
    
    fc7 = conv_bn_relu(filters = 1024, kernel_size = (1,1), stride= (2,2), padding='same',
                      name = {'conv':'conv_fc7', 'batch_norm':'bn_fc7','activation':'relu_fc7'})(fc6)
    
    conv8_1 = conv_bn_relu(filters = 256, kernel_size = (1,1),
                      name = {'conv':'conv8_1', 'batch_norm':'bn8_1','activation':'relu_8_1'})(fc7)
    
    conv8_2 = conv_bn_relu(filters = 512, kernel_size = (3,3), strides = (2,2),
                      name = {'conv':'conv8_2', 'batch_norm':'bn8_2','activation':'relu_8_2'})(conv8_1)
    
    # Block 7
    conv9_1 = conv_bn_relu(filters = 128, kernel_size = (1,1),
                      name = {'conv':'conv9_1', 'batch_norm':'bn9_1','activation':'relu_9_1'})(conv8_2)
    
    conv9_2 = conv_bn_relu(filters = 256, kernel_size = (3,3), strides = (2,2),
                      name = {'conv':'conv9_2', 'batch_norm':'bn9_2','activation':'relu_9_2'})(conv9_1)
    
    # Block 8
    conv10_1 = conv_bn_relu(filters = 128, kernel_size = (1,1),
                      name = {'conv':'conv10_1', 'batch_norm':'bn10_1','activation':'relu_10_1'})(conv9_2)
    
    
    
    conv10_2 = conv_bn_relu(filters = 256, kernel_size = (3,3), strides = (2,2), padding='same',
                      name = {'conv':'conv10_2', 'batch_norm':'bn10_2','activation':'relu_10_2'})(conv10_1)
    
    
    
    # Block 9
    conv11_1 = conv_bn_relu(filters = 128, 
                      name = {'conv':'conv11_1', 'batch_norm':'bn11_1','activation':'relu_11_1'})(conv10_2)
    
    conv11_2 = conv_bn_relu(filters = 256, kernel_size = (3,3), strides = (1,1),  padding='valid',
                      name = {'conv':'conv11_2', 'batch_norm':'b11_2','activation':'relu_11_2'})(conv11_1)
    
    # L2 normlaize layer
#    conv4_3_norm = Lambda(lambda x:K.l2_normalize(x, axis = -1), name = 'l2_normalization')(conv4_3)
    conv4_3_norm = L2Norm(name = 'l2_normalization')(conv4_3)
    
    # Calculate the spatial dimension of output layer
#    conv4_3_dim   = K.int_shape(conv4_3)[1:-1]
#    fc7_dim       = K.int_shape(fc7)[1:-1]
#    conv8_2_dim   = K.int_shape(conv8_2)[1:-1]
#    conv9_2_dim   = K.int_shape(conv9_2)[1:-1]
#    conv10_2_dim  = K.int_shape(conv10_2)[1:-1]
#    conv11_2_dim  = K.int_shape(conv11_2)[1:-1]
    
    
    # The class confidence score
    conv4_3_cls_score = Conv2D(filters = anchors[0] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(conv4_3_norm)
    
    fc7_cls_score     = Conv2D(filters = anchors[1] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(fc7)
    
    conv8_2_cls_score  = Conv2D(filters = anchors[2] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(conv8_2)
    
    conv9_2_cls_score  = Conv2D(filters = anchors[3] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(conv9_2)
    
    conv10_2_cls_score  = Conv2D(filters = anchors[4] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(conv10_2)
    
    conv11_2_cls_score  = Conv2D(filters = anchors[5] * num_classes, kernel_size=(3,3), activation = 'relu',
                               padding='same')(conv11_2)
    
    # Get the bounding box locations
    conv4_3_loc  = Conv2D(filters = anchors[0] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(conv4_3_norm)
    
    fc7_loc      = Conv2D(filters = anchors[1] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(fc7)
    
    conv8_2_loc  = Conv2D(filters = anchors[2] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(conv8_2)
    
    conv9_2_loc  = Conv2D(filters = anchors[3] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(conv9_2)
    
    conv10_2_loc  = Conv2D(filters = anchors[4] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(conv10_2)
    
    conv11_2_loc  = Conv2D(filters = anchors[5] * 4, kernel_size=(3,3), activation = 'linear',
                               padding='same')(conv11_2)
    
    
   
    # Reshape in 2D tensor
    
    conv4_3_cls_score   = Reshape(target_shape = (-1, num_classes))(conv4_3_cls_score)
    fc7_cls_score       = Reshape(target_shape = (-1, num_classes))(fc7_cls_score)
    conv8_2_cls_score   = Reshape(target_shape = (-1, num_classes))(conv8_2_cls_score)
    conv9_2_cls_score   = Reshape(target_shape = (-1, num_classes))(conv9_2_cls_score)
    conv10_2_cls_score  = Reshape(target_shape = (-1, num_classes))(conv10_2_cls_score)
    conv11_2_cls_score  = Reshape(target_shape = (-1, num_classes))(conv11_2_cls_score)
    
    conv4_3_loc         = Reshape(target_shape = (-1, 4))(conv4_3_loc)
    fc7_loc             = Reshape(target_shape = (-1, 4))(fc7_loc)
    conv8_2_loc         = Reshape(target_shape = (-1, 4))(conv8_2_loc)
    conv9_2_loc         = Reshape(target_shape = (-1, 4))(conv9_2_loc)
    conv10_2_loc        = Reshape(target_shape = (-1, 4))(conv10_2_loc)
    conv11_2_loc        = Reshape(target_shape = (-1, 4))(conv11_2_loc)
    
    cls_score = Concatenate(axis=1, name = 'classification_score')([conv4_3_cls_score, 
                                                                    fc7_cls_score,
                                                                    conv8_2_cls_score,
                                                                    conv9_2_cls_score,
                                                                    conv10_2_cls_score,
                                                                    conv11_2_cls_score])
    
    cls_score = Activation('softmax')(cls_score)
    
    loc      = Concatenate(axis = 1, name = 'regression_layer')([conv4_3_loc,
                                                                  fc7_loc,
                                                                  conv8_2_loc,
                                                                  conv9_2_loc,
                                                                  conv10_2_loc,
                                                                  conv11_2_loc])
    
    # Note add L2 Normalization
    
    output = Concatenate(name = 'final_layer')([loc, cls_score])
    
    return Model(inputs = input_, outputs = output)



if __name__ == '__main__':
#    with tf.device('/cpu:0'):
    model = SSD300(input_shape = (300, 300, 3))
#        print(model.get_weights())
    #    model.load_weights('VGG_ILSVRC_16_layers_fc_reduced.h5', by_name = True)
    #    print(model.get_weights())
    model.summary()