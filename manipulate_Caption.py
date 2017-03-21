#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################
import os
import matplotlib 
matplotlib.use('Agg')    
import matplotlib.pyplot as plt
import sys
import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import gzip
from PIL import Image
from numpy import math
#theano.config.optimizer = 'None'
#theano.config.exeption_verbosity = 'low'
#theano.config.compute_test_value = 'warn'
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer,Deconv2DLayer,DenseLayer
    print('cuda available')
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer,Deconv2DLayer,DenseLayer ,Deconv2DLayer
########
#from lasagne.layers import Conv2DLayer, MaxPool2DLayer,DenseLayer,Deconv2DLayer
theano.config.floatX = 'float32'
try:
   import cPickle as pickle
except:
   import pickle

stop_words =   ['a','an','the','of']  
############################################################################
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
####################################################################






class transform_Caption():
    
    def __init__(self,size_F):
        
        self.size_F = size_F
        #self.Feature_Kernel = np.zeros((I0,5*size_F))
        
        
         
    def Build_Vector(self,X0,Y0,X064by64,X,Index_Image):
        
        Feature_Kernel = np.zeros((Index_Image, 5*self.size_F))
        V_X = X0
        V_Y= Y0
        V_X64by640 = X064by64
        # suppression nan 
        # Ajust_Data  = Input.dropna()
        # fusion toute les descriptions et combine all to have a corpus
        Corpus_Train = np.array([''.join(Doc) for Doc in X])
        # count and normilize the corpus
        #Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(1,1), stop_words = 'english', strip_accents='unicode',norm='12')
        #Vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words = stop_words )
        Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(2,1),stop_words= stop_words)
        
        #Vectorizer = text.HashingVectorizer(stop_words = stop_words,non_negative=True, n_features = 2 ** 8)
        Count_Train_Vector = Vectorizer.fit_transform(Corpus_Train)
        
        
        
        
       
                
        # projection dim 4 ou alors calcul imilitude direct 2000 comme random simulation
        LS_Model = TruncatedSVD(n_components = 5,random_state=2000)
        LS_Model.fit(Count_Train_Vector)
        # Nouveau vecteur avec dimension reduit
        Vect_Proj = LS_Model.fit_transform(Count_Train_Vector,y=None)
        #Vect_Proj0 = [x for xs in Vect_Proj for x in xs]
        #print(len(Corpus_Train),len(Vect_Proj0),Vect_Proj0)
        
        #print(Vect_Proj[1],np.asarray(Vect_Proj[1]))
      
        #si je veux
        Temp1 = pd.DataFrame(Vect_Proj, columns = ['dim1','dim2','dim3','dim4','dim5'])
        Temp2 = pd.DataFrame(Vect_Proj[:,0],columns = ['Cos_Sim'])
        Temp1.index = range(0,Index_Image)
        Temp2.index = range(0,Index_Image)
        #print(Temp1.iloc[1])
        
        # project the 64-dimensional data to a lower dimension


        
        
        for J in range(0,Index_Image):
            Temp = Temp1.iloc[J:J+1]
            temp = pd.DataFrame(cosine_similarity(Temp.values,Temp1.values))
            Temp2['Cos_Sim'] = temp.transpose()
            Temp2 = Temp2.sort_values(by=['Cos_Sim'])
            #print(Temp2.index[-2],Temp2.index[-3])
            #print(Temp2.iloc[-2])
            # fusionner avec le prmier, deux premiers, 5 premiers et exploiter extraction feature ds autoencodeur
            # vecteur de taille 5 ou on ajoute des elts a V_X_Carre_train[:,:Index_Image+1,:,:,:]
            #  8*Index_Image*3*32*32, Index_Image : nombre image echantilllon d'entrainement 0 a 3 0 a 32 : repartition en 4 de image initiale ; 4 a 7, 16 a 48
            V_X[8:15,J,:,:,:] = X0[0:7,Temp2.index[-2],:,:,:]
            V_Y[8:15,J,:,:,:] = Y0[0:7,Temp2.index[-2],:,:,:]
            V_X[16:23,J,:,:,:] = X0[0:7,Temp2.index[-3],:,:,:]
            V_Y[16:23,J,:,:,:] = Y0[0:7,Temp2.index[-3],:,:,:]
            
            
            
            # Quelques feature pour l'image 64 by 64 avec 0 au Centre
            
            Nbre = 0
            params = {'bandwidth': np.logspace(-1, 1,20)}
            grid = GridSearchCV(KernelDensity(), params)
            temp = np.reshape(V_X64by640[J,:,:,:],(64,192)) # 3*64*64
            
            temp /= (255/2)
            temp -= 1
            
            pca = PCA(n_components = 5,svd_solver='full')
            data = pca.fit_transform(temp)
            grid.fit(data)
            # use the best estimator to compute the kernel density estimate
            kde = grid.best_estimator_
            new_data = kde.sample(self.size_F//5,random_state=2000)
            new_data -=new_data.min()
            new_data /=(new_data.max()-new_data.min())
            #new_data += 1
            new_data *= (255)
            new_data = np.uint8(new_data)
            new_data = new_data.astype('float32') 
            Feature_Kernel[J,Nbre:Nbre + self.size_F] = new_data.reshape(1,-1)
            Nbre =Nbre+self.size_F 
            
            
            
            
            
            
            # project the 3*32*32 -dimensional data to a lower dimension
            # use grid search cross-validation to optimize the bandwidth
            
            for K in(0,4):
                params = {'bandwidth': np.logspace(-1, 1,20)}
                grid = GridSearchCV(KernelDensity(), params)
                temp = np.reshape(V_X[K,Temp2.index[-1],:,:,:],(32,96)) # un quart du tableau 3*32*32 separation initiale 0*32-0*32; 32*64-0*32; 32*64-0*32; 32*64-32*64
                temp /= (255/2)
                
                temp -= 1
                pca = PCA(n_components = 5,svd_solver='full')
                data = pca.fit_transform(temp)
                grid.fit(data)
                # use the best estimator to compute the kernel density estimate
                kde = grid.best_estimator_
                new_data = kde.sample(self.size_F//5,random_state=2000)
                #new_data += 1
                new_data *= (255)
                new_data = np.uint8(new_data)
                new_data = new_data.astype('float32') 
                #print(np.sum(pca.explained_variance_ratio_)) 
                #print(new_data)
                # sample 44 new points from the data 2000 comme simule dristribution
                #new_data = pca.inverse_transform(new_data)
                Feature_Kernel[J,Nbre:Nbre + self.size_F] = new_data.reshape(1,-1)
                Nbre = Nbre + self.size_F 
            
            
            
            
        return  V_X, V_Y, Feature_Kernel
    

    
    
    
    