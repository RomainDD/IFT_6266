#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:43:55 2017

@author: djoumbid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:44:02 2017

@author: djoumbid
"""
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
# 5 meilleurs images et psca pour extraire facteur
# projeter texte ds un espace de dim 10 et consider coe feature input   
############################################################################
#frImportation des Classes
from discriminator_model import discriminator
from generative_model import generative
from manipulate_Data_Benchmark_Final1 import manipulate
#from simulate_distribution import distribution

#from autoencoder import autoencoder
# from manipulate_Caption import transform_Caption
# le texte en effet donne plusieurs descrition de l'image
############################################################################
# Lecture des donnees via la classe manipulate.data : on a un vect de int de shape : Nbre_Image*Couleur*64*64
# un vecteur Y simulation du centre Nbre_Image*3*32*32 et une liste contenant description textuelle de l'image

# Pour chaque carre on retient le future high level au niveau local 4 carre et 48 feauture ie de dimension 4*3*4*4  = 192
Size_F_per_M_G = 5 #  on projete en dim 5 et *20  on 100 
Size_F_per_M_L = 3 #  on local sur projete en dim 3 et *20  on 100 
size_Feature_Kernel_G = 5*Size_F_per_M_G  # on  va gerer le global feature plus tard car tres gros pour identifier similitude ,  GLOBAL et LOCAL on veut 100 on a Size_F_per_M feature pour 3*64*64 GLOBAL et Size_F_per_M feature pour chacun LOCAL des 4 blocs de 3*32*32
size_Feature_Kernel_L = 5*Size_F_per_M_L  
size_Feature_Autocodeur_L = 128 # 
size_Feature_Autocodeur_G = 32#
size_Feature_Caption = 16
size_Feature =   1*size_Feature_Autocodeur_L + 1*size_Feature_Autocodeur_G + 1*size_Feature_Caption # en effet 100 pour 4 Blocs locaux et 1 bloc global 

BATCH_SIZE = 128
BATCH_SIZE_Enc = 1
size_distribution =100
type_LvsG =2  ##  1- GLOBAL  sur 3*64*64; 2: LOCAL sur 3*32*32
Size_Epoch = 100
Size_Encode_G = 3*4*4  #  3*8*8  192 Facteurs
Size_Encode_L = 3*4*4  #  3*4*4   192 Facteurs.....
############################################################################
batch_idx_train =0
batch_idx_val =0
train_size = 200*129      #88  10*135 10 epoch pour 128 image par minibatch 138 car dans un set on se dit que l'on exclu 7 qui peuvent etre en noir et blanc
valid_size =  100*129    #44 77 5*135 3 epoch pour 32 image par minibatch

image_path ="inpainting/"    # Fichier des image
split_train ="train2014"
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
split_valid ="val2014"
###################################################
Ind_supp_train = 6782  # les premiers servent a entrainer l"autoencodeur marge je ne comprends pourquoi les first data sont a 1 
Ind_supp_val = 2504     # les premiers servent a entrainer l"autoencodeur
#### Simulation Distribution train et val pour le Reseau G
type_model_sim = 1# 1- normal noise  2-  3- 
type_nonlinear =1        # 1 Relu 2- tanh 3- sigmoid
type_iniParam = 2        # 1 HeNormal 2- Normal 3- Normal
## Construction Batch  pour Train et Validation toujours avec la classe manipulate.data
Manipulate = manipulate(BATCH_SIZE,size_Feature,Size_Epoch)
##################################################################
Distribution_train    = "Distribution_train.out"
Distribution_val      = "Distribution_val.out"

Feature_train_AutoEncode         = "V_Feature_train_AutoEncode_Final2.out"
Feature_val_AutoEncode            = "V_Feature_val_AutoEncode_Final2.out"

Feature_train_Caption         = "V_Feature_train_Caption_Final.out"
Feature_val_Caption           = "V_Feature_val_Caption_Final.out"



###########################################################################
train_data = manipulate(train_size,size_Feature,Size_Epoch)
X_train,Y_train32by32,Size_train,V_Noise_train,V_Feature_train_AE, V_Feature_train_Caption = train_data.Get_Data(batch_idx_train,image_path,split_train,caption_path,Distribution_train,Feature_train_AutoEncode,Feature_train_Caption,Ind_supp_train)
####################



valid_data = manipulate(valid_size,size_Feature,Size_Epoch)
X_val,Y_val32by32,Size_val,V_Noise_val,V_Feature_val_AE, V_Feature_val_Caption  = valid_data.Get_Data(batch_idx_val,image_path,split_valid,caption_path,Distribution_val,Feature_val_AutoEncode,Feature_val_Caption,Ind_supp_val)


Image_Test_PR = Y_val32by32[0:2,:,:,:].astype('float32')
Feat_and_Noise   = np.c_[V_Noise_val,V_Feature_val_AE, V_Feature_val_Caption]
Feat_and_Noise_Test_PR = Feat_and_Noise[0:BATCH_SIZE,:].astype('float32') 
##################
bath_size_Sim_t = Y_train32by32.shape[0]
bath_size_Sim_v = Y_val32by32.shape[0]
N_BATCHES = X_train.shape[0] // BATCH_SIZE
N_VAL_BATCHES = X_val.shape[0] // BATCH_SIZE
##################
nbatch = 0
idx = np.random.permutation(X_train.shape[0])
temp = idx[nbatch*BATCH_SIZE:(nbatch+1)*BATCH_SIZE]
#X_d64by64,Y_d32by32,Temp_Noise,Feature_AE,Feature_Caption = Manipulate.batch_gen(X_train,Y_train32by32,V_Noise_train,V_Feature_train_AE,V_Feature_train_Caption,temp)
##X_d64by64,Y_d32by32,Temp_Noise,Feature_AE,Feature_Caption   = Manipulate.batch_gen(X_val,Y_val32by32,V_Noise_val,V_Feature_val_AE,V_Feature_val_Caption,temp)

#Aranger partout ou j"ai train_D et Val en fainsant temp = idx[nbatch*BATCH_SIZE:(nbatch+1)*BATCH_SIZE] et un appel de Manipulate.batch_gen 

##############################################################################
# la variable categorielle prend 1 sur le vrai Ytrain de l'input X_train = 32*32
# et 0 sur le 32*32 simulee en fonction de Xtrain
Y_train_true_Binary = np.ones(shape=(Y_train32by32.shape[0])).astype('float32')
Y_train_gene_Binary1 = np.ones(shape=(Y_train32by32.shape[0])).astype('float32')
Y_train_gene_Binary0 = np.zeros(shape=(Y_train32by32.shape[0])).astype('float32')
Y_val_true_Binary   = np.ones(shape=(Y_val32by32.shape[0])).astype('float32')
Y_val_gene_Binary0   = np.zeros(shape=(Y_val32by32.shape[0])).astype('float32')
Y_val_gene_Binary1   = np.ones(shape=(Y_val32by32.shape[0])).astype('float32')
######################################################################
#Utiliser Autoencodeur pour extraire des features Locaux et globauX
# recontruire l'image 3*64*64 en remplacant le centre 32*32 par Y_3*32*32 (vraie centre)
# Separer en 4 pour Local (vertical, horizontal, diagonal bas, diagonal haut et entier pour global
## Description symbole et variable theano pour initialiser les fonction theano servant au calcul de la perte 
#### true data 32*32 with  Y = vector of 1 and Generate Data with  Y = vector of 0
X_d = T.tensor4()
Y_d = T.ivector()
X_g = T.tensor4()
Y_g = T.ivector()
Z_random = T.fmatrix()
Feature = T.fmatrix()
Y_g_reduit = T.fmatrix()# on passe de i*3*64*64 a i*(3*64*64)
X1Q1D= T.tensor3()
X1Q1G= T.tensor3()
X1Q2D= T.tensor3()
X1Q2G= T.tensor3()
X1Q3D= T.tensor3()
X1Q3G= T.tensor3()
X1Q4D= T.tensor3()
X1Q4G= T.tensor3()
X2Q1D= T.tensor3()
X2Q1G= T.tensor3()
X2Q2D= T.tensor3()
X2Q2G= T.tensor3()
X2Q3D= T.tensor3()
X2Q3G= T.tensor3()
X2Q4D= T.tensor3()
X2Q4G= T.tensor3()
X3Q1D= T.tensor3()
X3Q1G= T.tensor3()
X3Q2D= T.tensor3()
X3Q2G= T.tensor3()
X3Q3D= T.tensor3()
X3Q3G= T.tensor3()
X3Q4D= T.tensor3()
X3Q4G= T.tensor3()
FeatD= T.fmatrix()
FeatG= T.fmatrix()
L_d = T.scalar()
#################################################################################
## Construire un echantillon par regroupement texte sempblable ou reshapinig
##### n effet on a 4 carre par image, si forte ressemblance critere texte on merge les 4 images de chacun pour le test on prend param de lènsemble + proche
###################################################################
#### Construction de l'Architecture pour Discriminator et Init Parameters
size_input = (Y_train32by32.shape[0],Y_train32by32.shape[1],Y_train32by32.shape[2],Y_train32by32.shape[3])
Discriminator =  discriminator(X_d,size_input,type_nonlinear,type_iniParam)
Discriminator.d_architecture()
output_D_Train = lasagne.layers.get_output(Discriminator.architecture[-1], X_d, deterministic=False)
Feat_Discr_before_last = lasagne.layers.get_output(Discriminator.architecture[-2],X_d,deterministic=False)
########################## Construction de l'Architecture pour Genarator et Init Parameters
size_dist = (BATCH_SIZE,size_distribution+size_Feature)
Generator =  generative(BATCH_SIZE,size_dist,Z_random)
Generator.g_architecture()
Generate_X = lasagne.layers.get_output(Generator.architecture[-1],Z_random,deterministic=False)
output_G_Train = lasagne.layers.get_output(Discriminator.architecture[-1], Generate_X, deterministic=False)
output_D_Train  = lasagne.layers.get_output(Discriminator.architecture[-1], X_d, deterministic=False)
#################################################################################
#### Le Discrininator sait que echantillon simule est essentiellement simule voir avec 1 coe ds le papier
####  Calcul de la fonction de perte via binary entropy voir papier lecun prevision Video
loss_penality0 = T.mean(sqError(X1Q1D,X1Q1G))+ T.mean(sqError(X1Q2D,X1Q2G))+T.mean(sqError(X1Q3D,X1Q3G))+T.mean(sqError(X1Q4D,X1Q4G))+T.mean(sqError(X2Q1D,X2Q1G))+ T.mean(sqError(X2Q2D,X2Q2G))+T.mean(sqError(X2Q3D,X2Q3G))+T.mean(sqError(X2Q4D,X2Q4G))+T.mean(sqError(X3Q1D,X3Q1G))+ T.mean(sqError(X3Q2D,X3Q2G))+T.mean(sqError(X3Q3D,X3Q3G))+T.mean(sqError(X3Q4D,X3Q4G))
loss_penality1 =T.mean(sqError(FeatD,FeatG))
loss_penality = 0.5*loss_penality1 + 0.5*loss_penality0
Peak_Signal_Noise_Ratio = 10*T.log10((T.square(T.max(X_g).astype('float32'))/T.mean(T.square(X_d-X_g).astype('float32'))))
################################################## Train 
# mean and Peak are the traditionl measure, add structural similarity. take account interaction and and window (look Wikipedia)
temp_D  =  - T.mean(T.log(output_D_Train)) 
temp_G  =  - T.mean(T.log(1 - output_G_Train))
temp_G1 =  - 0.5*T.mean(T.log(output_G_Train))

acc_D0  = T.mean(output_D_Train > 0.5) 
acc_D1 =  T.mean(output_G_Train < 0.5)
acc_D = 0.5*(acc_D0 + acc_D1)
loss_G = temp_G1 + (loss_penality)  
loss_D = 0.5*temp_D + 0.5*temp_G 
#loss = [loss_D,acc_D,loss_G,loss_penality0,loss_penality1,temp_G1,temp_D,Peak_Signal_Noise_Ratio]
loss = [loss_D,acc_D,loss_G,loss_penality0,loss_penality1,temp_G1,temp_D,Peak_Signal_Noise_Ratio]

################################################# Validation
Generate_X_Val = lasagne.layers.get_output(Generator.architecture[-1],Z_random,deterministic=True)
output_G_Val  = lasagne.layers.get_output(Discriminator.architecture[-1], Generate_X_Val, deterministic=True)
output_D_Val  = lasagne.layers.get_output(Discriminator.architecture[-1], X_d, deterministic=True)
acc_D00  = T.mean(output_D_Val > 0.5) 
acc_D10 =  T.mean(output_G_Val < 0.5)
acc_D_Val = 0.5*(acc_D00 + acc_D10)
temp_D_Val  =  - T.mean(T.log(output_D_Val)) 
temp_G_Val  =  - T.mean(T.log(1 - output_G_Val))
temp_G1_Val =  - 0.5*T.mean(T.log(output_G_Val))
loss_G_Val = temp_G1_Val +(loss_penality)  #0.20
loss_D_Val = 0.5*temp_D_Val  + 0.5*temp_G_Val
loss_Val = [loss_D_Val,acc_D_Val,temp_G1_Val,loss_penality0,loss_penality1,Peak_Signal_Noise_Ratio]
###################################################
#### Liste des Parametres 
params_D_list = lasagne.layers.get_all_params(Discriminator.architecture,trainable=True)
params_G_list = lasagne.layers.get_all_params(Generator.architecture,trainable=True)
#################################################################################
#Discriminator_params_updates  = lasagne.updates.momentum(loss_D,params_D_list,learning_rate = 0.001,momentum = 0.9)
#Generator_params_updates      = lasagne.updates.momentum(loss_G,params_G_list,learning_rate = 0.0005,momentum = 0.6)
Discriminator_params_updates  = lasagne.updates.adam(loss_D,params_D_list,learning_rate = 0.001,beta1 = 0.9)
Generator_params_updates      = lasagne.updates.adam(loss_G,params_G_list,learning_rate = 0.0005,beta1 = 0.6)
##########################################################
#######################
#Disc_Gen_Params = [(p,a) for p,a in zip(params_D_list,params_G_list)]
#Disc_Gen_Params = [p for p in params_D_list]
# Fonction Theano pour train and validation
Discriminator_fn_before_last = theano.function([X_d],Feat_Discr_before_last)
generate_fn_X = theano.function([Z_random],Generate_X)
generate_fn_X_Val = theano.function([Z_random],Generate_X_Val)
Peak_Signal_Noise_Ratio_fn = theano.function([X_d,X_g],Peak_Signal_Noise_Ratio)

train_fn_D    = theano.function([X_d,Y_d,X_g,Y_g,Z_random,X1Q1D,X1Q1G,X1Q2D,X1Q2G,X1Q3D,X1Q3G,X1Q4D,X1Q4G,X2Q1D,X2Q1G,X2Q2D,X2Q2G,X2Q3D,X2Q3G,X2Q4D,X2Q4G,X3Q1D,X3Q1G,X3Q2D,X3Q2G,X3Q3D,X3Q3G,X3Q4D,X3Q4G,FeatD,FeatG],loss,updates = Discriminator_params_updates,on_unused_input='ignore')
val_fn_D      = theano.function([X_d,Y_d,X_g,Y_g,Z_random,X1Q1D,X1Q1G,X1Q2D,X1Q2G,X1Q3D,X1Q3G,X1Q4D,X1Q4G,X2Q1D,X2Q1G,X2Q2D,X2Q2G,X2Q3D,X2Q3G,X2Q4D,X2Q4G,X3Q1D,X3Q1G,X3Q2D,X3Q2G,X3Q3D,X3Q3G,X3Q4D,X3Q4G,FeatD,FeatG],loss_Val,on_unused_input='ignore')
train_fn_G    = theano.function([X_d,Y_d,X_g,Y_g,Z_random,X1Q1D,X1Q1G,X1Q2D,X1Q2G,X1Q3D,X1Q3G,X1Q4D,X1Q4G,X2Q1D,X2Q1G,X2Q2D,X2Q2G,X2Q3D,X2Q3G,X2Q4D,X2Q4G,X3Q1D,X3Q1G,X3Q2D,X3Q2G,X3Q3D,X3Q3G,X3Q4D,X3Q4G,FeatD,FeatG],loss,updates = Generator_params_updates,on_unused_input='ignore')
train_fn_update_D = theano.function([X_d,Y_d,X_g,Y_g,Z_random,X1Q1D,X1Q1G,X1Q2D,X1Q2G,X1Q3D,X1Q3G,X1Q4D,X1Q4G,X2Q1D,X2Q1G,X2Q2D,X2Q2G,X2Q3D,X2Q3G,X2Q4D,X2Q4G,X3Q1D,X3Q1G,X3Q2D,X3Q2G,X3Q3D,X3Q3G,X3Q4D,X3Q4G,FeatD,FeatG,L_d], outputs=None, updates=Discriminator_params_updates,on_unused_input='ignore')
train_fn_update_G = theano.function([X_d,Y_d,X_g,Y_g,Z_random,X1Q1D,X1Q1G,X1Q2D,X1Q2G,X1Q3D,X1Q3G,X1Q4D,X1Q4G,X2Q1D,X2Q1G,X2Q2D,X2Q2G,X2Q3D,X2Q3G,X2Q4D,X2Q4G,X3Q1D,X3Q1G,X3Q2D,X3Q2G,X3Q3D,X3Q3G,X3Q4D,X3Q4G,FeatD,FeatG,L_d], outputs=None, updates=Generator_params_updates,on_unused_input='ignore')

##
## Load Model
#full_path = '/home/djoumbid/IFT6266/Output/saved_models/generator_epoch_3.pkl'
#with open(full_path, 'rb') as f:
#    values = pickle.load(f)
#lasagne.layers.set_all_param_values(Generator.architecture,values)
#full_path = '/home/djoumbid/IFT6266/Output/saved_models/discrminator_epoch_3.pkl'
#with open(full_path, 'rb') as f:
#    values = pickle.load(f)
#lasagne.layers.set_all_param_values(Discriminator.architecture,values)


Epoch_train_loss_d = []
Epoch_train_acc_d = []
Epoch_train_loss_g = []
Epoch_train_acc_g = []
Epoch_val_loss_D = []
Epoch_val_acc_D = []
Epoch_val_loss_G = []
Epoch_Penalty_Feat =[]
Epoch_Penalty_Real =[]
Epoch_PNR =[]

S_epoch = 30
for epoch in range(S_epoch):
####  Train discriminator with Fixed weight of Generator model
    End_Train_Set = False
    train_loss_d = []
    train_acc_d = []
    train_loss_g = []
    train_acc_g = []
    val_loss_D = []
    val_acc_D = []
    val_loss_G = []
    Penalty_Feat = []
    Penalty_Real = []
    
    
    while(End_Train_Set == False):
       
        
        
        nbatch = 0
        idx = np.random.permutation(X_train.shape[0])
        for nbatch in range(N_BATCHES):
            
            #X_d64by64,Y_d32by32,Temp_Noise,Feature_AE,Feature_Caption= next(train_D_batches)
            temp = idx[nbatch*BATCH_SIZE:(nbatch+1)*BATCH_SIZE]
            X_d64by64,Y_d32by32,Temp_Noise,Feature_AE,Feature_Caption = Manipulate.batch_gen(X_train,Y_train32by32,V_Noise_train,V_Feature_train_AE,V_Feature_train_Caption,temp)




            
            Y_dbinary1 = np.ones((BATCH_SIZE)).astype('int32')
            #Feature_G = Feature_G/(255/2)
            #Feature_G = Feature_G - 1
            Feature_AE = Feature_AE/(255/2)
            Feature_AE = Feature_AE- 1
#            Feature_AE = Feature_AE/(255)
            Feat_and_Noise   = np.c_[Temp_Noise,Feature_AE,Feature_Caption]
            target = generate_fn_X(Feat_and_Noise)  # dans -1 1
            Y_g32by32 = target
            #Y_g32by32 = target[:,:,16:48,16:48]
            #Y_g32by32 = np.uint8(Y_g32by32)
            Y_g32by32 = Y_g32by32.astype('float32') 
            ################################################################################
#            Y_g32by32 = Y_g32by32 + 1
#            Y_g32by32 = Y_g32by32*(255/2)
            Y_d32by32_V = Y_d32by32
            Y_d32by32 = Y_d32by32/(255/2)
            Y_d32by32 = Y_d32by32 - 1
#            Y_d32by32 = Y_d32by32/(255)
            
   
    
    
    
            #Y_d32by32 = np.uint8(Y_d32by32)
            Y_d32by32 = Y_d32by32.astype('float32') 
            Y_gbinary0 = np.zeros((BATCH_SIZE)).astype('int32')
            
            C1Q1G = Y_g32by32[:,0,12:20,12:20]
            C1Q2G = Y_g32by32[:,0,8:24,8:24]
            C1Q3G = Y_g32by32[:,0,4:28,4:28]
            C1Q4G = Y_g32by32[:,0,0:32,0:32]
            ###
            C2Q1G = Y_g32by32[:,1,12:20,12:20]
            C2Q2G = Y_g32by32[:,1,8:24,8:24]
            C2Q3G = Y_g32by32[:,1,4:28,4:28]
            C2Q4G = Y_g32by32[:,1,0:32,0:32]
            ##
            C3Q1G = Y_g32by32[:,2,12:20,12:20]
            C3Q2G = Y_g32by32[:,2,8:24,8:24]
            C3Q3G = Y_g32by32[:,2,4:28,4:28]
            C3Q4G = Y_g32by32[:,2,0:32,0:32]
            ##
            C1Q1D = Y_d32by32[:,0,12:20,12:20]
            C1Q2D = Y_d32by32[:,0,8:24,8:24]
            C1Q3D = Y_d32by32[:,0,4:28,4:28]
            C1Q4D = Y_d32by32[:,0,0:32,0:32]
            ##
            C2Q1D = Y_d32by32[:,1,12:20,12:20]
            C2Q2D = Y_d32by32[:,1,8:24,8:24]
            C2Q3D = Y_d32by32[:,1,4:28,4:28]
            C2Q4D = Y_d32by32[:,1,0:32,0:32]
            ##
            C3Q1D = Y_d32by32[:,2,12:20,12:20]
            C3Q2D = Y_d32by32[:,2,8:24,8:24]
            C3Q3D = Y_d32by32[:,2,4:28,4:28]
            C3Q4D = Y_d32by32[:,2,0:32,0:32]
            
            
            
            feat_before_g = Discriminator_fn_before_last(Y_g32by32)        
            feat_before_d = Discriminator_fn_before_last(Y_d32by32)
            #feat_before_g = Y_g32by32[1,0,0:32,0:32]
            #feat_before_d = Y_d32by32[1,0,0:32,0:32]
#            ###########################
            # Update D
            loss = train_fn_D(Y_d32by32,Y_dbinary1,Y_g32by32,Y_gbinary0,Feat_and_Noise,C1Q1D,C1Q1G,C1Q2D,C1Q2G,C1Q3D,C1Q3G,C1Q4D,C1Q4G,C2Q1D,C2Q1G,C2Q2D,C2Q2G,C2Q3D,C2Q3G,C2Q4D,C2Q4G,C3Q1D,C3Q1G,C3Q2D,C3Q2G,C3Q3D,C3Q3G,C3Q4D,C3Q4G,feat_before_d,feat_before_g)
            train_loss_d.append(loss[0])
            train_acc_d.append(loss[1])
            #Param2 = lasagne.layers.get_all_params(Discriminator.architecture[-1])
            #print(Param2[0],Param2[1],Param2[0].get_value(),Param2[1].get_value())
           ###########################
            # Update G % on fait coincider les 3 couleurs voir si les segments aussi fait la difference ajouter bocs carree 4*4 8*8 16*16 et 32*32
            #Y_g32by32 = generate_fn_X_true(Feat_and_Noise)
            #Y_g32by32 = Y_g32by32 + 1
            #Y_g32by32 = Y_g32by32*(255/2)
            loss= train_fn_G(Y_d32by32,Y_dbinary1,Y_g32by32,Y_dbinary1,Feat_and_Noise,C1Q1D,C1Q1G,C1Q2D,C1Q2G,C1Q3D,C1Q3G,C1Q4D,C1Q4G,C2Q1D,C2Q1G,C2Q2D,C2Q2G,C2Q3D,C2Q3G,C2Q4D,C2Q4G,C3Q1D,C3Q1G,C3Q2D,C3Q2G,C3Q3D,C3Q3G,C3Q4D,C3Q4G,feat_before_d,feat_before_g)
            train_loss_g.append(loss[5])
            train_acc_g.append(loss[2])
            #print(Y_d32by32.max(),Y_d32by32.min(),Y_g32by32.max(),Y_g32by32.min(),Feature_AE.max(),Feature_AE.min())
     ######################################################################################
       
        nbatch = 0
        idx = np.random.permutation(X_val.shape[0])
        nload = 1
        for nbatch in range(N_VAL_BATCHES):
            #X_d64by64,Y_d32by32,Temp_Noise, Feature_AE, Feature_Caption = next(val_D_batches)
            temp = idx[nbatch*BATCH_SIZE:(nbatch+1)*BATCH_SIZE]
            X_d64by64,Y_d32by32,Temp_Noise,Feature_AE,Feature_Caption = Manipulate.batch_gen(X_val,Y_val32by32,V_Noise_val,V_Feature_val_AE,V_Feature_val_Caption,temp)

            Y_dbinary1 = np.ones((BATCH_SIZE)).astype('int32')
            #Feature_G = Feature_G/(255/2)
            #Feature_G = Feature_G - 1
            Feature_AE = Feature_AE/(255/2)
            Feature_AE = Feature_AE- 1
#            Feature_AE = Feature_AE/(255)
           
            
           #Feat_and_Noise   = np.c_[Temp_Noise,Feature_AE,Feature_G]
            Feat_and_Noise   = np.c_[Temp_Noise,Feature_AE,Feature_Caption]
            ##Feat_and_Noise   = Temp_Noise
            #Y_g32by32 = generate_fn_X_Val(Feat_and_Noise)  # dans -1 1
            target = generate_fn_X_Val(Feat_and_Noise)  # dans -1 1
            Y_g32by32 = target
            #Y_g32by32 = target[:,:,16:48,16:48]
            #Y_g32by32 = np.uint8(Y_g32by32)
            Y_g32by32 = Y_g32by32.astype('float32') 
            ################################################################################
#            Y_g32by32 = Y_g32by32 + 1
#            Y_g32by32 = Y_g32by32*(255/2)
            
            Y_d32by32_V = Y_d32by32
            Y_d32by32 = Y_d32by32/(255/2)
            Y_d32by32 = Y_d32by32 - 1
#            Y_d32by32 = Y_d32by32/(255)
           
            
            #Y_d32by32 = np.uint8(Y_d32by32)
            Y_d32by32 = Y_d32by32.astype('float32') 
            Y_gbinary0 = np.zeros((BATCH_SIZE)).astype('int32')
            #print(Y_d32by32.max(),Y_g32by32.max(),Feature_G.max(),Feature_AE.max())
            #print(Y_d32by32.min(),Y_g32by32.min(),Feature_G.min(),Feature_AE.min())
            #########################################
            feat_before_g = Discriminator_fn_before_last(Y_g32by32)        
            feat_before_d = Discriminator_fn_before_last(Y_d32by32)
            #feat_before_g = Y_g32by32[1,0,0:32,0:32]
            #feat_before_d = Y_d32by32[1,0,0:32,0:32]
            
            
            C1Q1G = Y_g32by32[:,0,12:20,12:20]
            C1Q2G = Y_g32by32[:,0,8:24,8:24]
            C1Q3G = Y_g32by32[:,0,4:28,4:28]
            C1Q4G = Y_g32by32[:,0,0:32,0:32]
            ###
            C2Q1G = Y_g32by32[:,1,12:20,12:20]
            C2Q2G = Y_g32by32[:,1,8:24,8:24]
            C2Q3G = Y_g32by32[:,1,4:28,4:28]
            C2Q4G = Y_g32by32[:,1,0:32,0:32]
            ##
            C3Q1G = Y_g32by32[:,2,12:20,12:20]
            C3Q2G = Y_g32by32[:,2,8:24,8:24]
            C3Q3G = Y_g32by32[:,2,4:28,4:28]
            C3Q4G = Y_g32by32[:,2,0:32,0:32]
            ##
            C1Q1D = Y_d32by32[:,0,12:20,12:20]
            C1Q2D = Y_d32by32[:,0,8:24,8:24]
            C1Q3D = Y_d32by32[:,0,4:28,4:28]
            C1Q4D = Y_d32by32[:,0,0:32,0:32]
            ##
            C2Q1D = Y_d32by32[:,1,12:20,12:20]
            C2Q2D = Y_d32by32[:,1,8:24,8:24]
            C2Q3D = Y_d32by32[:,1,4:28,4:28]
            C2Q4D = Y_d32by32[:,1,0:32,0:32]
            ##
            C3Q1D = Y_d32by32[:,2,12:20,12:20]
            C3Q2D = Y_d32by32[:,2,8:24,8:24]
            C3Q3D = Y_d32by32[:,2,4:28,4:28]
            C3Q4D = Y_d32by32[:,2,0:32,0:32]
    
            loss_val = val_fn_D(Y_d32by32,Y_dbinary1,Y_g32by32,Y_gbinary0,Feat_and_Noise,C1Q1D,C1Q1G,C1Q2D,C1Q2G,C1Q3D,C1Q3G,C1Q4D,C1Q4G,C2Q1D,C2Q1G,C2Q2D,C2Q2G,C2Q3D,C2Q3G,C2Q4D,C2Q4G,C3Q1D,C3Q1G,C3Q2D,C3Q2G,C3Q3D,C3Q3G,C3Q4D,C3Q4G,feat_before_d,feat_before_g)
            
            val_loss_D.append(loss_val[0])
            val_acc_D.append(loss_val[1])
            val_loss_G.append(loss_val[2])
            Penalty_Feat.append(np.float(loss_val[3]))
            Penalty_Real.append(np.float(loss_val[4]))
        ##P_Signal_Noise_R = Peak_Signal_Noise_Ratio_fn(Y_d32by32[0:1],Y_g32by32[0:1])
        ########################################################################
        #print(Y_d32by32.max(),Y_d32by32.min(),Y_g32by32.max(),Y_g32by32.min(),Feature_AE.max(),Feature_AE.min())
        print(np.mean(np.asarray(Penalty_Feat)),np.mean(np.asarray(train_loss_d)),np.mean(np.asarray(train_loss_g)))
        ##########################################################################
        if (batch_idx_val*valid_size >= 38699):
            
            Temp1 = np.mean(np.asarray(train_loss_d)) 
            Epoch_train_loss_d.append(Temp1)
            Temp2 = np.mean(np.asarray(train_acc_d)) 
            Epoch_train_acc_d.append(Temp2)
            Temp3 = np.mean(np.asarray(train_loss_g)) 
            Epoch_train_loss_g.append(Temp3)
            Temp4 = np.mean(np.asarray(train_acc_g)) 
            Epoch_train_acc_g.append(Temp4)
            
            Temp5 = np.mean(np.asarray(val_loss_D)) 
            Epoch_val_loss_D.append( np.float(Temp5))
            Temp6 = np.mean(np.asarray(val_acc_D)) 
            Epoch_val_acc_D.append( np.float(Temp6))
            Temp7= np.mean(np.asarray(val_loss_G)) 
            Epoch_val_loss_G.append( np.float(Temp7))
            
            Temp8= np.mean(np.asarray(Penalty_Feat)) 
            Epoch_Penalty_Feat.append( np.float(Temp8))
            Temp9= np.mean(np.asarray(Penalty_Real)) 
            Epoch_Penalty_Real.append( np.float(Temp9))
            
            ####
            target = generate_fn_X_Val(Feat_and_Noise_Test_PR)  # dans -1 1
            #Y_g32by32_1 = target[0:2,:,16:48,16:48]
            Y_g32by32_1 = target
            Y_g32by32_1 = Y_g32by32.astype('float32') 
            Temp = Y_g32by32_1 + 1
            Temp = Temp*(255/2)
            P_Signal_Noise_R6 = np.float(Peak_Signal_Noise_Ratio_fn(Image_Test_PR[0:1,:,:,:],Temp[0:1,:,:,:]))
            
            
            Epoch_PNR.append(P_Signal_Noise_R6)
            manipulate.save_model(Discriminator.architecture, 'discrminator_epoch_%s.pkl' % epoch)
            manipulate.save_model(Generator.architecture, 'generator_epoch_%s.pkl' % epoch)
            print('Epoch {}, Train_loss {:.12f}    Valid_Loss {:.12f}           Train_loss G {:.03f}             val_loss_G {:.03f}'.format(epoch, Temp1, Temp5, Temp3, Temp7))
            print('          Train_Acc  {:.03f}           Valid_Acc  {:.03f}                    penalty1_feature {:.08f}   penalty0_Yd_Yg  {:.08f}  P_Signal_Noise_R6 {:.03f}   '.format(100*Temp2,100*Temp6,Temp9,Temp8,P_Signal_Noise_R6))
            
            batch_idx_train = 0
            train_data = manipulate(train_size,size_Feature,Size_Epoch)
            X_train,Y_train32by32,Size_train,V_Noise_train,V_Feature_train_AE, V_Feature_train_Caption = train_data.Get_Data(batch_idx_train,image_path,split_train,caption_path,Distribution_train,Feature_train_AutoEncode,Feature_train_Caption,Ind_supp_train)
            ###################
            batch_idx_val = 0
            valid_data = manipulate(valid_size,size_Feature,Size_Epoch)
            X_val,Y_val32by32,Size_val,V_Noise_val,V_Feature_val_AE, V_Feature_val_Caption  = valid_data.Get_Data(batch_idx_val,image_path,split_valid,caption_path,Distribution_val,Feature_val_AutoEncode,Feature_val_Caption,Ind_supp_val)
          
            ##################
            N_BATCHES = X_train.shape[0] // BATCH_SIZE
            N_VAL_BATCHES = X_val.shape[0] // BATCH_SIZE
            ##################
#            train_D_batches = Manipulate.batch_gen(X_train,Y_train32by32,V_Noise_train,V_Feature_train_AE,V_Feature_train_Caption,X_train.shape[0])
#            val_D_batches   = Manipulate.batch_gen(X_val,Y_val32by32,V_Noise_val,V_Feature_val_AE,V_Feature_val_Caption,X_val.shape[0])
           
            End_Train_Set = True
                 
        else :     
            batch_idx_train = batch_idx_train + 1
            train_data = manipulate(train_size,size_Feature,Size_Epoch)
            X_train,Y_train32by32,Size_train,V_Noise_train,V_Feature_train_AE, V_Feature_train_Caption = train_data.Get_Data(batch_idx_train,image_path,split_train,caption_path,Distribution_train,Feature_train_AutoEncode,Feature_train_Caption,Ind_supp_train)
            ###################
            batch_idx_val = batch_idx_val + 1
            valid_data = manipulate(valid_size,size_Feature,Size_Epoch)
            X_val,Y_val32by32,Size_val,V_Noise_val,V_Feature_val_AE, V_Feature_val_Caption  = valid_data.Get_Data(batch_idx_val,image_path,split_valid,caption_path,Distribution_val,Feature_val_AutoEncode,Feature_val_Caption,Ind_supp_val)
          
            ##################
            N_BATCHES = X_train.shape[0] // BATCH_SIZE
            N_VAL_BATCHES = X_val.shape[0] // BATCH_SIZE
            ##################
#            train_D_batches = Manipulate.batch_gen(X_train,Y_train32by32,V_Noise_train,V_Feature_train_AE,V_Feature_train_Caption,X_train.shape[0])
#            val_D_batches   = Manipulate.batch_gen(X_val,Y_val32by32,V_Noise_val,V_Feature_val_AE,V_Feature_val_Caption,X_val.shape[0])
#        
        
        ############################################################################## 
    ### Fin While 

    if (epoch == S_epoch-1):
        Y_g32by32 = Y_g32by32 + 1
        Y_g32by32 = Y_g32by32*(255/2)
        #Y_g32by32 = Y_g32by32*(255)
        Y_g32by32 = np.uint8(Y_g32by32)
        
        
        
        Y_d32by32 = Y_d32by32_V
        ########################################################
        P_Signal_Noise_R1 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[127:128],Y_g32by32[127:128])
        P_Signal_Noise_R2 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[120:121],Y_g32by32[120:121])
        P_Signal_Noise_R3 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[80:81],Y_g32by32[80:81])
        P_Signal_Noise_R4 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[60:61],Y_g32by32[60:61])
        P_Signal_Noise_R5 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[40:41],Y_g32by32[40:41])
        P_Signal_Noise_R6 = Peak_Signal_Noise_Ratio_fn(Y_d32by32[10:11],Y_g32by32[10:11])
       #Y_g32by32 = generate_fn_X_Val(Feat_and_Noise[119])
        T1=2
        T2=8
        plt.figure(figsize=(T2,T1))
        Manipulate.plot_img(Y_d32by32[10],Y_g32by32[10],0,T1,T2)
        Manipulate.plot_img(Y_d32by32[20],Y_g32by32[20],1,T1,T2)
        Manipulate.plot_img(Y_d32by32[30],Y_g32by32[30],2,T1,T2)
        Manipulate.plot_img(Y_d32by32[40],Y_g32by32[40],3,T1,T2)
        Manipulate.plot_img(Y_d32by32[50],Y_g32by32[50],4,T1,T2)
        Manipulate.plot_img(Y_d32by32[60],Y_g32by32[60],5,T1,T2)
        Manipulate.plot_img(Y_d32by32[70],Y_g32by32[70],6,T1,T2)
        Manipulate.plot_img(Y_d32by32[80],Y_g32by32[80],7,T1,T2)
        print(P_Signal_Noise_R1,P_Signal_Noise_R2,P_Signal_Noise_R3,P_Signal_Noise_R4,P_Signal_Noise_R5,P_Signal_Noise_R6)
        
      


#np.savetxt('Epoch_Penalty_Feature_M25',Epoch_Penalty_Feat)
#np.savetxt('Epoch_Penalty_Real_Image_M25',Epoch_Penalty_Real)
#####
#np.savetxt('Epoch_train_loss_d_M25',Epoch_train_loss_d)  
#np.savetxt('Epoch_train_loss_g_M25',Epoch_train_loss_g)  
#np.savetxt('Epoch_train_acc_d_M25',Epoch_train_acc_d)  
#np.savetxt('Epoch_train_acc_g_M25',Epoch_train_acc_g)  
#####
#np.savetxt('Epoch_val_loss_d_M25',Epoch_val_loss_D)  
#np.savetxt('Epoch_val_loss_g_M25',Epoch_val_loss_G)  
#np.savetxt('Epoch_val_acc_d_M25',Epoch_val_acc_D)
#np.savetxt('Epoch_Signal_Noise_M25',Epoch_PNR)


