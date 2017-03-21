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
    
    def __init__(self, batch_size,size_Feature):
        self.batch_size = batch_size
        self.size_Feature = size_Feature
        
    def batch_gen(self,X,Y,size):
        while True:
            idx = np.random.choice(size,self.batch_size)
            #V_idx[j] = idx
            #j = j+1
            yield X[idx].astype('float32'),Y[idx].astype('float32'),idx
        
         ### PATH need to be fixed
    def Get_Data(self,batch_idx,mscoco,split,caption_path):
        '''  Show an example of how to read the dataset   '''
        V_input  = np.zeros((self.batch_size,3,64,64))
        V_target = np.zeros((self.batch_size,3,32,32))
        V_target_64by64 = np.zeros((self.batch_size,3,64,64))
        V_Y_Carre = np.zeros((24,self.batch_size,3,32,32))
        V_X_Carre = np.zeros((24,self.batch_size,3,32,32))
        V_caption_dict = []
        x = T.tensor3()
        f = theano.function([x],outputs=x.dimshuffle(2,0,1))
        #m = T.ftensor3()
        #m_new = T.stack(T.concatenate([V_input[0],m]))
        #f = theano.function([V_input,m], outputs=[m_new])
        data_path = os.path.join(mscoco,split)
        caption_path = os.path.join(mscoco,caption_path)
        with open(caption_path,'rb') as fd:
            caption_dict = pickle.load(fd)
        
            #print (data_path + "/*.jpg")
        imgs = glob.glob(data_path + "/*.jpg")
        batch_imgs = imgs[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        j = 0
        for i, img_path in enumerate(batch_imgs):
            #print(i)
            img = Image.open(img_path)
            img_array = np.array(img)
            Input     = np.array(img)
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))  
            cap_id = os.path.basename(img_path)[:-4]
            #k=0 
            ### Get input/target from the images true immage 64*64, image with mask 64*64  true center 32*32
            
            if len(img_array.shape) == 3:
                V_target_64by64[j,:,:,:] = f(img_array)
                Input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                V_input[j,:,:,:] = f(Input) 
               
               #####
                
                    
                ##############################################################################
                target0 = img_array[center[0]-32:center[0], center[1] - 32:center[1], :]
                V_Y_Carre[0,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-32:center[0], center[1] - 32:center[1], :]
                V_X_Carre[0,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]-32:center[0], center[1]:center[1]+32, :]
                V_Y_Carre[1,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-32:center[0], center[1]:center[1]+32, :]
                V_X_Carre[1,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]:center[0]+32, center[1] - 32:center[1], :]
                V_Y_Carre[2,j,:,:,:] = f(target0)
                target0 =     Input[center[0]:center[0]+32, center[1] - 32:center[1], :]
                V_X_Carre[2,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]:center[0]+32, center[1]:center[1]+32, :]
                V_Y_Carre[3,j,:,:,:] = f(target0)
                target0 =     Input[center[0]:center[0]+32, center[1]:center[1]+32, :]
                V_X_Carre[3,j,:,:,:] = f(target0)
                ### ### ### ###
                target0 = img_array[center[0]-32+8:center[0]+8, center[1] - 32+8:center[1]+8, :]
                V_Y_Carre[4,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-32+8:center[0]+8, center[1] - 32+8:center[1]+8, :]
                V_X_Carre[4,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]-32+8:center[0]+8, center[1]-8:center[1]+32-8, :]
                V_Y_Carre[5,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-32+8:center[0]+8, center[1]-8:center[1]+32-8, :]
                V_X_Carre[5,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]-8:center[0]+32-8, center[1] - 32+8:center[1]+8, :]
                V_Y_Carre[6,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-8:center[0]+32-8, center[1] - 32+8:center[1]+8, :]
                V_X_Carre[6,j,:,:,:] = f(target0)
                ###
                target0 = img_array[center[0]-8:center[0]+32-8, center[1]-8:center[1]+32-8, :]
                V_Y_Carre[7,j,:,:,:] = f(target0)
                target0 =     Input[center[0]-8:center[0]+32-8, center[1]-8:center[1]+32-8, :]
                V_X_Carre[7,j,:,:,:] = f(target0)
                               
                ##############################################################################
                ###vtrue immage 64*64, image with mask 64*64  true center 32*32
                 #input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
               
                V_target[j,:,:,:] = f(target)
                V_caption_dict.append(caption_dict[cap_id])
                j = j+1  
                #print (i,cap_id,caption_dict[cap_id])  
                #print(input)
                #plt.imshow(input)  
                #eeee
            
            
            
            
        V_Temp_I = V_input        
        if (V_Temp_I.shape[0]!=j):
            
            V_input = V_input[:j,:,:,:]
            V_target = V_target[:j,:,:,:]  
            V_target_64by64 = V_target_64by64[:j,:,:,:]  
            
            V_Y_Carre = V_Y_Carre[:,:j,:,:,:] 
            V_X_Carre = V_X_Carre[:,:j,:,:,:]
            V_Temp_I = V_Temp_I[:j,:,:,:]
        Index_Image  = j  
        #print("fin first Party",V_input[0],Index_Image)     
            #else:
            #    try:
            #        input = np.copy(img_array)
            #        input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            #        target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
            #    except:
            #        pass
           
            
            #Image.fromarray(img_array).show()
            #Image.fromarray(input).show()
            #Image.fromarray(target).show()
       ###################################################################  Identification des description identiques
       # on a 100 Feature pour chacun des 4 blocs de 3*3232 et 100 pour le blocs masque 3*64*64
        Feature_Kernel = np.zeros((Index_Image, 5*self.size_Feature))
        # suppression nan 
        # Ajust_Data  = Input.dropna()
        # fusion toute les descriptions et combine all to have a corpus
        Corpus_Train = np.array([''.join(Doc) for Doc in V_caption_dict])
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
        Temp1 = pd.DataFrame(Vect_Proj, columns = ['dim1','dim2','dim3','dim4','dim5'])
        Temp2 = pd.DataFrame(Vect_Proj[:,0],columns = ['Cos_Sim'])
        Temp1.index = range(0,Index_Image)
        Temp2.index = range(0,Index_Image)
        #############################################  Creation Feature
        for ind in range(0,Index_Image):
            Temp = Temp1.iloc[ind:ind+1]
            temp = pd.DataFrame(cosine_similarity(Temp.values,Temp1.values))
            Temp2['Cos_Sim'] = temp.transpose()
            Temp2 = Temp2.sort_values(by=['Cos_Sim'])
            #print(Temp2.index[-2],Temp2.index[-3])
            #print(Temp2.iloc[-2])
            # fusionner avec le prmier, deux premiers, 5 premiers et exploiter extraction feature ds autoencodeur
            # vecteur de taille 5 ou on ajoute des elts a V_X_Carre_train[:,:Index_Image+1,:,:,:]
            #  8*Index_Image*3*32*32, Index_Image : nombre image echantilllon d'entrainement 0 a 3 0 a 32 : repartition en 4 de image initiale ; 4 a 7, 16 a 48
            V_X_Carre[8:15,ind,:,:,:] = V_X_Carre[0:7,Temp2.index[-2],:,:,:]
            V_Y_Carre[8:15,ind,:,:,:] = V_Y_Carre[0:7,Temp2.index[-2],:,:,:]
            V_X_Carre[16:23,ind,:,:,:] = V_X_Carre[0:7,Temp2.index[-3],:,:,:]
            V_Y_Carre[16:23,ind,:,:,:] = V_Y_Carre[0:7,Temp2.index[-3],:,:,:]
            #print(J,Temp2.index[-2],Temp2.index[-3])
            
            
            # Quelques feature pour l'image 64 by 64 avec 0 au Centre
            Nbre = 0
            params = {'bandwidth': np.logspace(-1, 1,20)}
            grid = GridSearchCV(KernelDensity(), params)
            temp0 = np.reshape(V_Temp_I[ind,:,:,:],(64,192)) # 3*64*64
            temp0 = temp0/(255/2)
            temp0 = temp0-1
            #temp /= (255/2)
            #temp -= 1
                  
            n_components = 5
            pca = PCA(n_components = n_components, svd_solver='full')
            data = pca.fit_transform(temp0)
            grid.fit(data)
            # use the best estimator to compute the kernel density estimate
            kde = grid.best_estimator_
            new_data = kde.sample(self.size_Feature//n_components,random_state=2000)
            temp00 = new_data.min()
            temp01 = new_data.max()
            new_data = new_data - temp00
            new_data = new_data/(temp00-temp01)
            #new_data += 1
            new_data = new_data*255
            new_data = np.uint8(new_data)
            new_data = new_data.astype('float32') 
            Feature_Kernel[ind,Nbre:Nbre + self.size_Feature] = new_data.reshape(1,-1)
            Nbre =Nbre+self.size_Feature 
           
             
#           project the 3*32*32 -dimensional data to a lower dimension
#           use grid search cross-validation to optimize the bandwidth
            
            for K in(0,4):
                params = {'bandwidth': np.logspace(-1, 1,20)}
                grid = GridSearchCV(KernelDensity(), params)
                temp0 = np.reshape(V_X_Carre[K,Temp2.index[-1],:,:,:],(32,96)) # un quart du tableau 3*32*32 separation initiale 0*32-0*32; 32*64-0*32; 32*64-0*32; 32*64-32*64
               
                temp0 = temp0/(255/2)
                temp0 = temp0-1
                
                
                pca = PCA(n_components = 5,svd_solver='full')
                data = pca.fit_transform(temp0)
                grid.fit(data)
                # use the best estimator to compute the kernel density estimate
                kde = grid.best_estimator_
                new_data = kde.sample(self.size_Feature//5,random_state=2000)
                #new_data += 1
                temp00 = new_data.min()
                temp01 = new_data.max()
                new_data = new_data - temp00
                new_data = new_data/(temp00-temp01)
                new_data = new_data*255
                new_data = np.uint8(new_data)
                #print(np.sum(pca.explained_variance_ratio_)) 
                #print(new_data)
                # sample 44 new points from the data 2000 comme simule dristribution
                #new_data = pca.inverse_transform(new_data)
                Feature_Kernel[ind,Nbre:Nbre + self.size_Feature] = new_data.reshape(1,-1)
                Nbre = Nbre + self.size_Feature
            
            
            
            
        return V_input,V_target,V_target_64by64,V_caption_dict,Index_Image,V_X_Carre,V_Y_Carre,Feature_Kernel
               
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
        imgs = glob.glob(data_path+"/*.jpg")
    
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

   

    def plot_img(self,img,title):
        plt.figure()
        plt.imshow((np.transpose(np.uint8(img),(1, 2, 0))), interpolation='nearest')
        plt.title(title)
        #plt.axis('off')
        plt.tight_layout()

    
    

    
#if __name__ == '__main__':
    #pass
    #resize_mscoco()
    #show_examples(5,5)       
        
        
        
        
        
        
        
        
