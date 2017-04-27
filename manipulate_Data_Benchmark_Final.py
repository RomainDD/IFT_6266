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
import shutil
try:
   import cPickle as pickle
except:
   import pickle
   
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
####################################################################
#########################################################################################################################################

class manipulate:
    
    def __init__(self,batch_size,size_Feature,Size_Epoch):
        self.batch_size = batch_size
        self.size_Feature = size_Feature
        #self.Size_Encode_L =  Size_Encode_L
        self.Size_Epoch    =  Size_Epoch
        #self.BATCH_SIZE_Enc = BATCH_SIZE_Enc
        #self.type_LvsG      = type_LvsG
        #self.size_Feature_Kernel = size_Feature_Kernel
        #self.size_Feature_Autocodeur = size_Feature_Autocodeur
   
                 
    def batch_gen(self,X,Y,Z,T,V,idx):
            #while True:
            #idx = np.random.choice(X.shape[0],X.shape[0])
            #idx = np.random.permutation(X.shape[0])
            #print(X.shape[0],size,len(idx),idx)
            #yield X[idx].astype('float32'),Y[idx].astype('float32'),Z[idx].astype('float32'),T[idx].astype('float32'),V[idx].astype('float32')
            return X[idx].astype('float32'),Y[idx].astype('float32'),Z[idx].astype('float32'),T[idx].astype('float32'),V[idx].astype('float32')
    

    def Get_Data2(self,split,Ind_supp):
        '''  Show an example of how to read the dataset   '''
        ################################################################
        if (split ==  "train"):
            V_input    = np.load('/home/djoumbid/IFT6266/Input/V_X64by64_train_%s.npy'%Ind_supp)
            V_target =  np.load('/home/djoumbid/IFT6266/Input/V_Y32BY32_train_%s.npy'%Ind_supp)
            V_Noise    = np.load('/home/djoumbid/IFT6266/Input/Noise_train_%s.npy'%Ind_supp)
            V_Feature = np.load('/home/djoumbid/IFT6266/Input/V_Feature_train_AutoEncode_G_%s.npy'%Ind_supp)
            V_Feature_AutoEncode = np.load('/home/djoumbid/IFT6266/Input/V_Feature_train_AutoEncode_L_%s.npy'%Ind_supp)
            V_Feature_Caption = np.load('/home/djoumbid/IFT6266/Input/Caption_train_%s.npy'%Ind_supp)
            #print('/home/djoumbid/IFT6266/Input/V_Y32BY32_train_%s.npy'%Ind_supp) 
            Index_Image = len(V_input)
        elif (split ==  "valid"):
            V_input    = np.load('/home/djoumbid/IFT6266/Input/V_X64by64_valid_%s.npy'%Ind_supp)
            V_target =  np.load('/home/djoumbid/IFT6266/Input/V_Y32BY32_valid_%s.npy'%Ind_supp)
            V_Noise    = np.load('/home/djoumbid/IFT6266/Input/Noise_valid_%s.npy'%Ind_supp)
            V_Feature = np.load('/home/djoumbid/IFT6266/Input/V_Feature_valid_AutoEncode_G_%s.npy'%Ind_supp)
            V_Feature_AutoEncode = np.load('/home/djoumbid/IFT6266/Input/V_Feature_valid_AutoEncode_L_%s.npy'%Ind_supp)
            V_Feature_Caption = np.load('/home/djoumbid/IFT6266/Input/Caption_valid_%s.npy'%Ind_supp)
            Index_Image = len(V_input)
            #print('/home/djoumbid/IFT6266/Input/V_X64by64_valid_%s.npy'%Ind_supp)    
            
        
        return V_input,V_target,Index_Image,V_Noise,V_Feature,V_Feature_AutoEncode,V_Feature_Caption



    
         ### PATH need to be fixed
    def Get_Data(self,batch_idx,mscoco,split,caption_path,D_Noise,D_Feature_AutoEncode,D_Feature_Caption,Ind_supp):
        '''  Show an example of how to read the dataset   '''
        V_input  = np.zeros((self.batch_size,3,64,64))
        V_target = np.zeros((self.batch_size,3,64,64))
        #V_target_64by64 = np.zeros((self.batch_size,3,64,64))
        V_Noise   = np.zeros((self.batch_size,100))
        V_Feature_AutoEncode =  np.zeros((self.batch_size,160))
        V_Feature_Caption = np.zeros((self.batch_size,16))
        #V_caption_dict = []
        x = T.tensor3()
        f = theano.function([x],outputs=x.dimshuffle(2,0,1))
        ################################################################
        Noise    = np.loadtxt(D_Noise)
        Feature_AutoEncode = np.loadtxt(D_Feature_AutoEncode)
        Feature_Caption = np.loadtxt(D_Feature_Caption)
        ##############################################################
        data_path = os.path.join(mscoco,split)
        #caption_path = os.path.join(mscoco,caption_path)
        #with open(caption_path,'rb') as fd:
        #    caption_dict = pickle.load(fd)
        
        if (batch_idx == 0):
            #Debut = batch_idx*self.batch_size
            Debut =Ind_supp
        else:
            Debut= batch_idx*self.batch_size
        
        #print("Debut",Debut)
        imgs = sorted(glob.glob(data_path + "/*.jpg"))
        
        batch_imgs = imgs[Debut:(batch_idx+1)*self.batch_size]
        Temp_N =  Noise[Debut:(batch_idx+1)*self.batch_size]
        Temp_Feat_AuEn = Feature_AutoEncode[Debut:(batch_idx+1)*self.batch_size]
        Temp_Feat_Caption = Feature_Caption[Debut:(batch_idx+1)*self.batch_size]
        
        j = 0
        for i, img_path in enumerate(batch_imgs):
            #print(i)
            img = Image.open(img_path)
            img_array = np.array(img)
            Input     = np.array(img)
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
            #cap_id = os.path.basename(img_path)[:-4]
            #k=0 
            ### Get input/target from the images true immage 64*64, image with mask 64*64  true center 32*32
            
            if len(img_array.shape) == 3:
                #V_target_64by64[j,:,:,:] = f(img_array)
                Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                V_input[j,:,:,:] = f(Input) 
               
                #####
                ##############################################################################
                ###vtrue immage 64*64, image with mask 64*64  true center 32*32
                #target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                target = img_array
               
                V_target[j,:,:,:] = f(target)
                # V_caption_dict.append(caption_dict[cap_id])
                V_Noise[j,:] = Temp_N[i,:]
                V_Feature_AutoEncode[j,:] = Temp_Feat_AuEn[i,:]  # si train on commence a 1280  sinon 640
                V_Feature_Caption[j,:] =Temp_Feat_Caption[i,:] 
                #######################
                j = j+1  
                            
             
        if (V_input.shape[0]!=j):
            
            V_input = V_input[:j,:,:,:]
            V_target = V_target[:j,:,:,:]  
            #V_target_64by64 = V_target_64by64[:j,:,:,:]  
            
           
            V_Noise = V_Noise[:j,:]
            V_Feature_AutoEncode = V_Feature_AutoEncode[:j,:]
            V_Feature_Caption = V_Feature_Caption[:j,:] 
            
        Index_Image  = j  
        
        return V_input,V_target,Index_Image,V_Noise,V_Feature_AutoEncode,V_Feature_Caption
       
    def resize_mscoco():
        ''' function used to create the dataset,Resize original MS_COCO Image into 64x64 images '''
        ########################################################### 
        ### PATH need to be fixed
        data_path="/datasets/coco/coco/images/train2014"
        save_dir = "/Tmp/64_64/train2014/"
        ###########################################################  
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        preserve_ratio = True
        image_size = (64, 64)
        #crop_size = (32, 32)
        imgs = sorted(glob.glob(data_path+"/*.jpg"))
    
        for i, img_path in enumerate(imgs):
            img = Image.open(img_path)
            print (i,len(imgs),img_path)
            if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
                if not preserve_ratio:
                    img = img.resize((image_size),Image.ANTIALIAS)
                else:
                    ### Resize based on the smallest dimension
                    scale = image_size[0] / float(np.min(img.size))
                    new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                    img = img.resize((new_size), Image.ANTIALIAS)
                    ### Crop the 64/64 center
                    tocrop = np.array(img)
                    center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                    print (tocrop.shape,center,(center[0]-32,center[0]+32),(center[1]-32,center[1]+32))
                    if len(tocrop.shape) == 3:
                        tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                    else:
                        tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                    img = Image.fromarray(tocrop)
            img.save(save_dir + os.path.basename(img_path))

   

    def plot_img(self,imgR,imgG,i,T1,T2):
        ax1 = plt.subplot(T1,T2,i + 1)
        plt.imshow(np.transpose(np.uint8(imgR),(1, 2, 0)))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax = plt.subplot(T1,T2,i + T2 + 1)
        plt.imshow(np.transpose(np.uint8(imgG),(1, 2, 0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
             
         
         #        #fig = plt.figure()
#        ax = fig.add_subplot(T1,T2,i)
#        ax.imshow((np.transpose(np.uint8(imgR),(1, 2, 0))), interpolation='nearest')
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        ax = fig.add_subplot(T1,T2,i+1)
#        ax.imshow((np.transpose(np.uint8(imgG),(1, 2, 0))), interpolation='nearest')
#        #plt.axis('off')
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        #plt.title(title)
#        plt.tight_layout()

        
        
        
        

    def save_model(network, filename):
        """
        Saves the parameters of a model to a pkl file
        """
        path = '/home/djoumbid/IFT6266/Output/saved_models/'
    
        if not os.path.exists(path):
            os.makedirs(path)
    
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(lasagne.layers.get_all_param_values(network), f)
        print('Model saved to file %s' % filename)
        
    def dump_objects_output(args, object, filename):
        """
        Dumps any object using pickle into ./output/objects_dump/ directory
        """
        path = '/home/djoumbid/IFT6266/Output/saved_models/'
        if not os.path.exists(path):
            os.makedirs(path)
    
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(object, f)
        print ('Object saved to file %s' % filename)
         
        
    
#if __name__ == '__main__':
    #pass
    #resize_mscoco()
    #show_examples(5,5)       
        
        
        
        
        
        
        
        
