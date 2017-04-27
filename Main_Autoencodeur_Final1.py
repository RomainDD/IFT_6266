# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 06:07:58 2017

@author: David
"""
#####################################################################
import theano
import lasagne
import numpy as np
import os, sys
import glob

import numpy as np
import PIL.Image as Image
from skimage.transform import resize
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
theano.config.floatX = 'float32'
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
#import mnist,gzip
import matplotlib.pyplot as plt
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
from lasagne.layers.dnn  import Conv2DDNNLayer

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer,TransposedConv2DLayer,DenseLayer
    print('cuda available')
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer,TransposedConv2DLayer,DenseLayer
########
try:
   import cPickle as pickle
except:
   import _pickle as pickle
###############################################################################
#frImportation des Classes
# le texte en effet donne plusieurs descrition de l'image
import pandas as pd
import numpy as np
import os, io
from nltk import tokenize
############################################################################
# tokenizer = RegeexpTokenizer(r'barre oblique a cote de 7 W+')
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
# These_Versio0
#from nltk import stopwords
#from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
stop_words =   ['a','an','the','of'] 
from autoencoder_model_Final1 import autoencoder
from manipulate_Data_Benchmark_Final1 import manipulate
#################################################################### on autoencode sur les 1280 objets
#########################################################################################################################################
NB_TRAIN = 82782
NB_VALID = 40504

#Size_Epoch = 150
Size_Epoch =150

batch_idx =0
batch_size_train = 8782
batch_size_val = 3504

#Size_Encode = 16
#Size_filter =(32,32)


Size_Encode = 128+32
Size_filter =(3,3)

######
BATCH_SIZE_Auto_End = 32
image_path ="inpainting/"    # Fichier des image
split_train ="train2014"
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
split_valid ="val2014"
image_path ="inpainting/"    # Fichier des image
Len_train_Reste = NB_TRAIN-batch_size_train 
       
'''  Show an example of how to read the dataset   '''
V_input_train  = np.zeros((batch_size_train ,3,64,64))
V_target_64by64_train = np.zeros((batch_size_train,3,64,64))
V_input_val  = np.zeros((batch_size_val ,3,64,64))
V_target_64by64_val = np.zeros((batch_size_val,3,64,64))

#V_caption_dict = []
x = T.tensor3()
f = theano.function([x],outputs=x.dimshuffle(2,0,1))
data_path = os.path.join(image_path,split_train)
imgs = sorted(glob.glob(data_path + "/*.jpg"))
batch_imgs_train = imgs[batch_idx*batch_size_train:(batch_idx+1)*batch_size_train ]
j = 0
for i, img_path in enumerate(batch_imgs_train):
    #print(i)
    img = Image.open(img_path)
    img_array = np.array(img)
    Input     = np.array(img)
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
        
    if len(img_array.shape) == 3:
        V_target_64by64_train[j,:,:,:] = f(img_array)
        Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        V_input_train[j,:,:,:] = f(Input) 
        ##############################################################################
        ###vtrue immage 64*64, image with mask 64*64  true center 32*32
        #######################
        j = j+1  
if (V_input_train.shape[0]!=j):
    V_input_train = V_input_train[:j,:,:,:]
    V_target_64by64_train = V_target_64by64_train[:j,:,:,:]
    
    
    
################################################################

##################################################
x = T.tensor3()
f = theano.function([x],outputs=x.dimshuffle(2,0,1))
data_path = os.path.join(image_path,split_valid)
imgs = sorted(glob.glob(data_path + "/*.jpg"))
batch_imgs_val = imgs[batch_idx*batch_size_val:(batch_idx+1)*batch_size_val]
j = 0
for i, img_path in enumerate(batch_imgs_val):
    #print(i)
    img = Image.open(img_path)
    img_array = np.array(img)
    Input     = np.array(img)
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
        
    if len(img_array.shape) == 3:
        V_target_64by64_val[j,:,:,:] = f(img_array)
        Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        V_input_val[j,:,:,:] = f(Input) 
        #######################
        j = j+1  
if (V_input_val.shape[0]!=j):
    V_input_val = V_input_val[:j,:,:,:]
    V_target_64by64_val = V_target_64by64_val[:j,:,:,:]  
 


V_input = np.r_[V_input_train,V_input_val]
V_target_64by64 = np.r_[V_target_64by64_train,V_target_64by64_val]

###########################################################################################################        
# Autoencodeur avec 8 images on verra apres avec les images similaires via les captions
X_d = T.tensor4()
X_g = T.tensor4()
##size_input_Global = (V_input.shape[0],V_input.shape[1],V_input.shape[2],V_input.shape[3])
size_input  = (V_input.shape[0],V_input.shape[1],V_input.shape[2],V_input.shape[3])
#if (ind % 100 == 0):
#    print('Feature Creation for image :',ind)
Auto_encoder_local = autoencoder(size_input,Size_Encode,Size_Epoch,BATCH_SIZE_Auto_End,Size_filter)
Auto_encoder_local.build_network(X_d,X_g)
Epoch_train_loss_AE, Epoch_train_SN_Ratio_AE  =  Auto_encoder_local.learn(V_input,V_target_64by64)




np.savetxt('Epoch_train_loss_AE',Epoch_train_loss_AE)
np.savetxt('Epoch_Valid_SN_Ratio_AE',Epoch_train_SN_Ratio_AE)

###############################################################################################################################
V_Feature_train = np.zeros((NB_TRAIN-batch_size_train,Size_Encode))
data_path = os.path.join(image_path,split_train)
#imgs = glob.glob(data_path + "/*.jpg")
imgs = sorted(glob.glob(data_path + "/*.jpg"))
j_temp = 0
j = batch_size_train
for j in range(batch_size_train,NB_TRAIN):  
    img_path = imgs[j:j+1]
    for j0, img_path in enumerate(img_path):
        img = Image.open(img_path)
        img_array = np.array(img)
        Input     = np.array(img)
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
        #cap_id = os.path.basename(img_path)[:-4]
        if len(img_array.shape) == 3:
            Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            #print(Input,j0)     
            V_temp = f(Input) 
            V_temp = np.reshape(V_temp,(1,3,64,64))
            V_encodeL = Auto_encoder_local.encode_L((V_temp).astype('float32'))
            #print(np.size(V_encodeL))
            V_encodeL += 1
            V_encodeL *= (255/2)
            V_encodeL = np.uint8(V_encodeL)
            #####
            V_encodeG = Auto_encoder_local.encode_G((V_temp).astype('float32'))
            #print(np.size(V_encodeG))
            V_encodeG += 1
            V_encodeG *= (255/2)
            V_encodeG = np.uint8(V_encodeG)
            V_Feature_train[j_temp,:] = np.c_[V_encodeL,V_encodeG]               
                           
                         
                           
            #print(V_Feature_train[j,0])  85 Facteurs 25 Global 15*4 locaux
            if (j % 5000== 0):
                print(j)
        else: 
            V_Feature_train[j_temp,:] =  np.ones((1,Size_Encode))  
    j_temp = j_temp+1       
    
    
V_Feature_train = np.r_[np.ones((batch_size_train,160)),V_Feature_train]    
np.savetxt('V_Feature_train_AutoEncode_Final2.out',V_Feature_train)    
  


######################################################################################################################################
V_Feature_val = np.zeros((NB_VALID-batch_size_val,Size_Encode))
data_path = os.path.join(image_path,split_valid)
imgs = sorted(glob.glob(data_path + "/*.jpg"))
j_temp = 0
j = batch_size_val
for j in range(batch_size_val,NB_VALID):  
    img_path = imgs[j:j+1]
    for j0, img_path in enumerate(img_path):
        img = Image.open(img_path)
        img_array = np.array(img)
        Input     = np.array(img)
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
        #cap_id = os.path.basename(img_path)[:-4]
        if len(img_array.shape) == 3:
            Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            V_temp = f(Input) 
            V_temp = np.reshape(V_temp,(1,3,64,64))
            V_encodeL = Auto_encoder_local.encode_L((V_temp).astype('float32'))
            V_encodeL += 1
            V_encodeL *= (255/2)
            V_encodeL = np.uint8(V_encodeL)
            #####
            V_encodeG = Auto_encoder_local.encode_G((V_temp).astype('float32'))
            V_encodeG += 1
            V_encodeG *= (255/2)
            V_encodeG = np.uint8(V_encodeG)
            V_Feature_val[j_temp,:] = np.c_[V_encodeL,V_encodeG]                
                         
                         
                         
            #print(V_Feature_train[j,0])  85 Facteurs 25 Global 15*4 locaux
            if (j % 5000== 0):
                print(j)
        else: 
            V_Feature_val[j_temp,:] =  np.ones((1,Size_Encode))  
    j_temp = j_temp+1       
V_Feature_val = np.r_[np.ones((batch_size_val,160)),V_Feature_val]     
np.savetxt('V_Feature_val_AutoEncode_Final2.out',V_Feature_val)    
    
 # version initiale (5,5)       
        
        
        
