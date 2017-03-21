#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:04:03 2017

@author: djoumbid
"""
#####################################################################
import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
#######
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer,DenseLayer
    print('cuda available')
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer,DenseLayer
########
try:
   import cPickle as pickle
except:
   import pickle
   
###############################################################################
# fixed random seeds
class distribution():
    def __init__(self,type_model_sim,size_dist,bath_size_train,bath_size_val):
        # Prepare Theano variables for inputs and targets
        self.type_model_sim = type_model_sim
        self.size_dist  = size_dist
        self.bath_size_train  = bath_size_train
        self.bath_size_val  = bath_size_val
        # rng = np.random.RandomState(1)
        # self.theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        
        
    def Sim_Distribution(self):
        #rng_data = np.random.RandomState(4000)
        rng = np.random.RandomState(2000)
        theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
        
        temp1 = (self.bath_size_train,self.size_dist)
        temp2 = (self.bath_size_val,self.size_dist)
        if (self.type_model_sim==1):  # Normal  0 et 1 
           Distribution_train = theano_rng.uniform(low =-np.sqrt(3),high =np.sqrt(3),size=temp1,dtype='float32')
           Distribution_val   = theano_rng.uniform(low =-np.sqrt(3),high =np.sqrt(3),size=temp2,dtype='float32')
           
        elif(self.type_model_sim==2):
           Distribution_train = theano_rng.normal(0,1,size=temp1,dtype='float32')
           Distribution_val   = theano_rng.normal(0,1,size=temp2,dtype='float32') 
           
        else:     
            pass
        
        return  Distribution_train,Distribution_val  
        