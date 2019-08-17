#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:04:54 2019

@author: Purnendu Mishra
"""

import tensorflow as tf


from keras.backend.tensorflow_backend import set_session
##********************************************************
## For GPU
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.50
set_session(tf.Session(config=config))

#
##********************************************************

from keras import backend as K

from SSD_generate_anchors import generate_ssd_priors
from CustomDataLoader import DataAugmentor

from pathlib import Path

import collections
from ssd300_model import SSD300

from keras_ssd_loss import SSDLoss
#from ssd_loss_function_v2 import CustomLoss
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from CustomCallback import CSVLogger, PolyLR
import numpy as np

import argparse

tf_session = K.get_session()
#%%****************************************************************************
parser = argparse.ArgumentParser()

parser.add_argument('-b','--batch_size', default=8,  type=int, help='Batch size for training')
parser.add_argument('-e','--epochs',     default=120, type=int, help='number of epochs for training')

args  = parser.parse_args()


#%%****************************************************************************
#root                  = Path.home()/'data'/'VOCdevkit'/'VOC2007'
root                  = Path.home()/'Documents'/'DATASETS'/'VOCdevkit'

#%%****************************************************************************

exp  =  'EXP_01'.upper()

mean = np.array([114.02898, 107.86698,  99.73119], dtype=np.float32)
std  = np.array( [69.89365, 69.07726, 72.30074], dtype=np.float32)

target_size = (300,300)

batch_size  = args.batch_size

num_epochs  = args.epochs

initial_lr  = 1e-2

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
trainloader = DataAugmentor(rescale    = 1./255.0,
                            horizontal_flip=True,
                            )

valloader  = DataAugmentor(rescale    = 1./255.0,
                           )


train_generator   = trainloader.flow_from_directory(
                                            root        = root,
                                            data_folder = ['VOC2007','VOC2012'],
                                            target_size = target_size,
                                            batch_size  = batch_size,
                                            shuffle     = True,
                                            priors      = priors
                                            )

val_generator     = valloader.flow_from_directory (
                                            root         = root,
                                            data_folder = ['VOC2007','VOC2012'],
                                            target_size  = target_size,
                                            batch_size   = batch_size,
                                            shuffle      = False,
                                            priors       = priors
                                            )


steps_per_epoch  = len(train_generator)
validation_steps = len(val_generator)
#%*****************************************************************************
model = SSD300(input_shape=(300,300, 3), num_classes=21)

model.load_weights('VGG_ILSVRC_16_layers_fc_reduced.h5', by_name=True)
#model.summary()

loss_fn = SSDLoss()

model.compile(optimizer = SGD(lr= initial_lr, momentum = 0.9 , nesterov=True, decay=1e-5),
              loss      = loss_fn.compute_loss)
#%%****************************************************************************
experiment_name = '{}_SSD300_VOC_BS-{}_EP-{}'.format(exp, batch_size, num_epochs)

records           = Path.cwd()/'records'
checkpoint_path   = Path.cwd()/'checkpoints'

if not records.exists():
    records.mkdir()
    
if not checkpoint_path.exists():
    checkpoint_path.mkdir()

lr_scheulder = PolyLR(base_lr   = initial_lr, 
                      power     = 0.9, 
                      nb_epochs = num_epochs, 
                      steps_per_epoch = steps_per_epoch,
                      mode = None)

checkpoint = ModelCheckpoint(filepath          = '{}/{}_ChkPt_'.format(checkpoint_path, experiment_name) + '{val_loss:.4f}.hdf5',
                             monitor           = 'val_loss',
                             mode              = 'auto',
                             save_best_only    = True, 
                             save_weights_only = True,
                             period            = 5,
                             verbose           = 1)

csvlog = CSVLogger(records/'{}.csv'.format(experiment_name), separator=',', append=True)

callbacks = [csvlog, checkpoint, lr_scheulder]

#%%****************************************************************************
model.fit_generator(generator       =  train_generator,
                    validation_data =  val_generator,   
                    epochs          =  num_epochs,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps,
                    verbose   = 1,
                    use_multiprocessing = True,
                    callbacks           = callbacks,
                    workers = 4)

model.save_weights(filepath = 'FinalWeights_{}.h5'.format(experiment_name))
                 

