#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:29:52 2017

@author: djoumbid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:12:57 2017

@author: djoumbid
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:07:27 2017

@author: djoumbid
"""
# 1000 sur train et 5oo sur val pour un autoencoderur et utiliser pour encode environ 75 facteurs
####################################################################
################################# approche Autoencodeur avec 24 echantillon a base de mesure de proximite du texte (tif et hashing approach
#################################  A prroche avec PCA et extraction de 5 facteurs  
import theano
import lasagne

theano.config.floatX = 'float32'
try:
   import cPickle as pickle
except:
   import pickle
import os
import matplotlib 
matplotlib.use('Agg')    
import matplotlib.pyplot as plt
import sys

import numpy as np
from lasagne.objectives import squared_error as sqError
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import gzip
from PIL import Image
from numpy import math
#from fuel.schemes import ShuffledScheme, SequentialScheme
#from fuel.streams import DataStream
#from fuel.datasets.hdf5 import H5PYDataset

#theano.config.optimizer = 'None'
#theano.config.exeption_verbosity = 'low'
#theano.config.compute_test_value = 'warn'
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer, Deconv2DLayer, DenseLayer
    print('cuda available')
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer,Deconv2DLayer,DenseLayer ,Deconv2DLayer
########
#from lasagne.layers import Conv2DLayer, MaxPool2DLayer,DenseLayer,Deconv2DLayer
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

import theano
import lasagne
import numpy as np
import os, sys
import glob
#################################################################### 
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
  
############################################################################
#frImportation des Classes
from discriminator_model import discriminator
from generative_model import generative
from manipulate_Data_Benchmark_V2 import manipulate
from simulate_distribution import distribution

#from autoencoder import autoencoder
# from manipulate_Caption import transform_Caption
# le texte en effet donne plusieurs descrition de l'image
############################################################################
# Lecture des donnees via la classe manipulate.data : on a un vect de int de shape : Nbre_Image*Couleur*64*64
# un vecteur Y simulation du centre Nbre_Image*3*32*32 et une liste contenant description textuelle de l'image
NB_TRAIN = 82782
NB_VALID = 40504
V_input     = np.zeros((1,3,64,64))
V_input_4     = np.zeros((4,3,32,32))
mscoco ="inpainting/"    # Fichier des image
split ="train2014"
image_path ="inpainting/"    # Fichier des image
split_train ="train2014"
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
V_caption_dict = []
x = T.tensor3()
f = theano.function([x],outputs=x.dimshuffle(2,0,1))
data_path = os.path.join(mscoco,split)
caption_path = os.path.join(mscoco,caption_path)

with open(caption_path,'rb') as fd:
    caption_dict = pickle.load(fd)
imgs = sorted(glob.glob(data_path + "/*.jpg"))

# Calculer   distance text retenir les 5 plus proches et mettre ajouter dans l'echantillon pour extraire Facteurs Globaux
# s"arranger pour avoir juste 5 elets et donc pas beaucoup dans la RAM
batch_imgs = imgs[0:NB_TRAIN]
j = 0
for i, img_path in enumerate(batch_imgs):
    #print(i)
    img = Image.open(img_path)
    img_array = np.array(img)
    Input     = np.array(img)
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
    cap_id = os.path.basename(img_path)[:-4]
    V_caption_dict.append(caption_dict[cap_id])
    j = j+1  

Corpus_Train = np.array([''.join(Doc) for Doc in V_caption_dict])
# count and normilize the corpus
#Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(1,1), stop_words = 'english', strip_accents='unicode',norm='12')
#Vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words = stop_words )

############################# First
#Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(2,1),stop_words= stop_words)
#Count_Train_Vector = Vectorizer.fit_transform(Corpus_Train)
# projection dim 4 ou alors calcul imilitude direct 2000 comme random simulation
#LS_Model = TruncatedSVD(n_components = 5,random_state=2000)
###############################

####Second
#Vectorizer = text.HashingVectorizer(stop_words = stop_words,non_negative=True, n_features = 2 ** 8)
Vectorizer = text.TfidfVectorizer(sublinear_tf=True,ngram_range=(1,3), max_df=0.5,stop_words= stop_words)
Count_Train_Vector = Vectorizer.fit_transform(Corpus_Train)
#print("n_samples: %d, n_features: %d" % Count_Train_Vector.shape)
# projection dim 4 ou alors calcul imilitude direct 2000 comme random simulation
LS_Model = TruncatedSVD(n_components = 16,random_state=2000)
LS_Model.fit(Count_Train_Vector)
Vect_Proj_train = LS_Model.fit_transform(Count_Train_Vector,y=None)

#Temp1 = pd.DataFrame(Vect_Proj, columns = ['dim1','dim2','dim3','dim4','dim5'])
#Temp2 = pd.DataFrame(Vect_Proj[:,0],columns = ['Cos_Sim'])
#Temp1.index = range(0,Index_Image)
#Temp2.index = range(0,Index_Image)
#for J in range(0,Index_Image):
#    Temp = Temp1.iloc[J:J+1]
#    temp = pd.DataFrame(cosine_similarity(Temp.values,Temp1.values))
#    Temp2['Cos_Sim'] = temp.transpose()
#    Temp2 = Temp2.sort_values(by=['Cos_Sim'])

#print(Vect_Proj_train.min(),Vect_Proj_train.max())
np.savetxt('V_Feature_train_Caption_Final.out',Vect_Proj_train)    
######################################################################################################################################
# un vecteur Y simulation du centre Nbre_Image*3*32*32 et une liste contenant description textuelle de l'image
NB_VALID = 40504
V_input     = np.zeros((1,3,64,64))
V_input_4     = np.zeros((4,3,32,32))
mscoco ="inpainting/"    # Fichier des image
image_path ="inpainting/"    # Fichier des image
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
split="val2014"

V_caption_dict = []
x = T.tensor3()
f = theano.function([x],outputs=x.dimshuffle(2,0,1))
data_path = os.path.join(mscoco,split)
caption_path = os.path.join(mscoco,caption_path)

with open(caption_path,'rb') as fd:
    caption_dict = pickle.load(fd)
imgs = sorted(glob.glob(data_path + "/*.jpg"))


# Calculer   distance text retenir les 5 plus proches et mettre ajouter dans l'echantillon pour extraire Facteurs Globaux
# s"arranger pour avoir juste 5 elets et donc pas beaucoup dans la RAM
batch_imgs = imgs[0:NB_VALID]
j = 0
for i, img_path in enumerate(batch_imgs):
    #print(i)
    img = Image.open(img_path)
    img_array = np.array(img)
    Input     = np.array(img)
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
    cap_id = os.path.basename(img_path)[:-4]
    V_caption_dict.append(caption_dict[cap_id])
    j = j+1  

Corpus_Train = np.array([''.join(Doc) for Doc in V_caption_dict])
# count and normilize the corpus
#Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(1,1), stop_words = 'english', strip_accents='unicode',norm='12')
#Vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words = stop_words )

############################# First
#Vectorizer = text.TfidfVectorizer(min_df=1,ngram_range=(2,1),stop_words= stop_words)
#Count_Train_Vector = Vectorizer.fit_transform(Corpus_Train)
# projection dim 4 ou alors calcul imilitude direct 2000 comme random simulation
#LS_Model = TruncatedSVD(n_components = 5,random_state=2000)
###############################
####Second
Vectorizer = text.TfidfVectorizer(sublinear_tf=True,ngram_range=(1,3),max_df=0.5,stop_words= stop_words)
Count_Train_Vector = Vectorizer.fit_transform(Corpus_Train)
# projection dim 4 ou alors calcul imilitude direct 2000 comme random simulation
LS_Model = TruncatedSVD(n_components = 16,random_state=2000)

LS_Model.fit(Count_Train_Vector)
# Nouveau vecteur avec dimension reduit
Vect_Proj_val = LS_Model.fit_transform(Count_Train_Vector,y=None)   
    
np.savetxt('V_Feature_val_Caption_Final.out',Vect_Proj_val) 

print(Vect_Proj_train.min(),Vect_Proj_train.max(),Vect_Proj_val.min(),Vect_Proj_val.max())

   
    