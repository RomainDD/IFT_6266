#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:07:27 2017

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
#frImportation des Classes
from discriminator_model import discriminator
from generative_model import generative
from manipulate_Data import manipulate
from simulate_distribution import distribution
from autoencoder import autoencoder
# from manipulate_Caption import transform_Caption
# le texte en effet donne plusieurs descrition de l'image
############################################################################
# Lecture des donnees via la classe manipulate.data : on a un vect de int de shape : Nbre_Image*Couleur*64*64
# un vecteur Y simulation du centre Nbre_Image*3*32*32 et une liste contenant description textuelle de l'image
batch_idx =0
# Pour chaque carre on retient le future high level au niveau local 4 carre et 48 feauture ie de dimension 4*3*4*4  = 192
Size_F_per_M = 20
size_autoenco_tr = 24
size_Feature_Kernel = 5*Size_F_per_M  #  GLOBAL et LOCAL on veut 100 on a Size_F_per_M feature pour 3*64*64 GLOBAL et Size_F_per_M feature pour chacun LOCAL des 4 blocs de 3*32*32
size_Feature_Autocodeur = size_autoenco_tr*3*8*8  # LOCAL Autoencodeur sur 3 images X_3*64*64 les 3 plus proches que l'on separe en 4 blocs puis autoencodeur pour extraire 3*8*8 Feauture
size_Feature = size_Feature_Kernel + size_Feature_Autocodeur 
type_LvsG =1  ##  1- GLOBAL  sur 3*64*64; 2: LOCAL sur 3*32*32
Size_Epoch = 60
Size_Encode_G = 3*8*8  #  3*8*8  192 Facteurs
Size_Encode_L = 3*8*8  #  3*4*4   192 Facteurs.....
BATCH_SIZE_Enc = 2
size_distribution =100


train_size = 2000
train_data = manipulate(train_size,size_Feature_Kernel)
image_path ="inpainting/"    # Fichier des image
split_train ="train2014"
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
X_train,Y_train32by32,Y_train64by64,Caption_dict_train,Size_train,V_X_Carre_train,V_Y_Carre_train,V_Feat_Kernel_train   = train_data.Get_Data(batch_idx,image_path,split_train,caption_path)
###################
valid_size =  500
valid_data = manipulate(valid_size,size_Feature_Kernel)
split_valid ="val2014"
X_val,Y_val32by32,Y_val64by64,Caption_dict_val,Size_val,V_X_Carre_val,V_Y_Carre_val,V_Feat_Kernel_val = valid_data.Get_Data(batch_idx,image_path,split_valid,caption_path)
##############################################################################
# la variable categorielle prend 1 sur le vrai Ytrain de l'input X_train = 32*32
# et 0 sur le 32*32 simulee en fonction de Xtrain
Y_train_true_Binary = np.ones(shape=(Y_train32by32.shape[0])).astype('float32')
Y_train_gene_Binary1 = np.ones(shape=(Y_train32by32.shape[0])).astype('float32')
Y_train_gene_Binary0 = np.zeros(shape=(Y_train32by32.shape[0])).astype('float32')
Y_val_true_Binary   = np.ones(shape=(Y_val32by32.shape[0])).astype('float32')
Y_val_gene_Binary0   = np.zeros(shape=(Y_val32by32.shape[0])).astype('float32')
Y_val_gene_Binary1   = np.ones(shape=(Y_val32by32.shape[0])).astype('float32')
####################################################################
# Construction Batch  pour Train et Validation toujours avec la classe manipulate.data
BATCH_SIZE = 5
Manipulate = manipulate(BATCH_SIZE,size_Feature_Kernel)
N_BATCHES = X_train.shape[0] // BATCH_SIZE
N_VAL_BATCHES = X_val.shape[0] // BATCH_SIZE
train_D_batches = Manipulate.batch_gen(X_train,Y_train32by32,X_train.shape[0])
val_D_batches   = Manipulate.batch_gen(X_val,Y_val32by32,X_val.shape[0])
#### Simulation Distribution train et val pour le Reseau G
type_model_sim = 1   # 1- normal noise  2-  3- 
type_nonlinear =1        # 1 Relu 2- tanh 3- sigmoid
type_iniParam = 2        # 1 HeNormal 2- Normal 3- Normal
bath_size_Sim_t = Y_train32by32.shape[0]
bath_size_Sim_v = Y_val32by32.shape[0]
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
#################################################################################
## Construire un echantillon par regroupement texte sempblable ou reshapinig
# imput X_train, Y_train64by64,
size_input_Global = (X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])
size_input_Local = (Y_train32by32.shape[0],Y_train32by32.shape[1],Y_train32by32.shape[2],Y_train32by32.shape[3])
##################################################################################
#Auto_encoder = autoencoder(size_input_Global,Size_Encode_G,Size_Epoch,BATCH_SIZE_Enc,X_train,Y_train64by64,type_LvsG)
#Auto_encoder.build_network(X_d,X_g)
#Auto_encoder.learn(X_train,Y_train64by64)
#################################################################################
##### n effet on a 4 carre par image, si forte ressemblance critere texte on merge les 4 images de chacun pour le test on prend param de lènsemble + proche
# Pour chaque carre on retient le future high level au niveau local 4 carre et 48 feauture ie de dimension 4*3*4*4  = 192
#Temp =  np.transpose((np.uint8(V_Y_Carre_train[2,1,:,:,:])),(1, 2, 0)) 
#plt.figure()
#plt.axis('off')
#plt.imshow(Temp,interpolation='nearest')
#plt.show()
#################################################################################
##### n effet on a 4 carre par image, si forte ressemblance critere texte on merge les 4 images de chacun pour le test on prend param de lènsemble + proche
# Pour chaque carre on retient le future high level au niveau local 4 carre et 48 feauture ie de dimension 4*3*4*4  = 192
Sample_Feature_train = np.zeros((X_train.shape[0],size_Feature_Autocodeur))
Sample_Feature_val = np.zeros((X_val.shape[0],size_Feature_Autocodeur))

type_LvsG =2  ##  Local
for Rang_Image in range(Size_train):
    if (Rang_Image % 50 == 0):
        print('Feature Creation for image :',Rang_Image)
    Index_Image =Rang_Image
    #X_Sub_Image = V_X_Carre_train[:,0:Index_Image+1,:,:,:]
    #Y_Sub_Image = V_Y_Carre_train[:,0:Index_Image+1,:,:,:]
    Auto_encoder_local = autoencoder(size_input_Local,Size_Encode_L,Size_Epoch,BATCH_SIZE_Enc,V_X_Carre_train[:,Index_Image,:,:,:],V_Y_Carre_train[:,Index_Image,:,:,:],type_LvsG,size_Feature_Kernel)
    Auto_encoder_local.build_network(X_d,X_g)
    Auto_encoder_local.learn(V_X_Carre_train[:,Index_Image,:,:,:],V_Y_Carre_train[:,Index_Image,:,:,:])
    V_encode = Auto_encoder_local.encode((V_X_Carre_train[:,Index_Image,:,:,:]).astype('float32'))
    V_encode = np.reshape(V_encode,size_Feature_Autocodeur)
    #V_encode += 1
    #V_encode *= (255/2)
    #V_encode = np.uint8(V_encode)
    # on laisse la valeur entre -1 et 1
    Sample_Feature_train[Index_Image,:] = V_encode
                        
for Rang_Image in range(Size_val):
    if (Rang_Image % 50 == 0):
        print('Feature Creation for image :',Rang_Image)
    Index_Image =Rang_Image
    #X_Sub_Image = V_X_Carre_train[:,0:Index_Image+1,:,:,:]
    #Y_Sub_Image = V_Y_Carre_train[:,0:Index_Image+1,:,:,:]
    Auto_encoder_local = autoencoder(size_input_Local,Size_Encode_L,Size_Epoch,BATCH_SIZE_Enc,V_X_Carre_train[:,Index_Image,:,:,:],V_Y_Carre_train[:,Index_Image,:,:,:],type_LvsG,size_Feature_Kernel)
    Auto_encoder_local.build_network(X_d,X_g)
    Auto_encoder_local.learn(V_X_Carre_train[:,Index_Image,:,:,:],V_Y_Carre_train[:,Index_Image,:,:,:])
    V_encode = Auto_encoder_local.encode((V_X_Carre_train[:,Index_Image,:,:,:]).astype('float32'))
    V_encode = np.reshape(V_encode,size_Feature_Autocodeur)
    #V_encode += 1
    #V_encode *= (255/2)
    #V_encode = np.uint8(V_encode)
    # on laisse la valeur entre -1 et 1
    Sample_Feature_val[Index_Image,:] = V_encode
                      
                      
np.savetxt('Sample_Feature_val.out' ,Sample_Feature_val) 
np.savetxt('Sample_Feature_train.out',Sample_Feature_train)
np.savetxt('V_Feat_Kernel_train.out',V_Feat_Kernel_train) 
np.savetxt('V_Feat_Kernel_val.out',V_Feat_Kernel_val)
#np.savetxt('Sample_Feature_val.out',(Sample_Feature_val,Sample_Feature_train))               
ggggggggggg
###################################################################
# Simulation Noise via distribution uniforme pour le modele generatif aavec la classe simulate_distribution 
# Construction Batch  pour Train et Validation toujours avec avec la classe manipulate.data
Sim_train_Input = distribution(type_model_sim,size_distribution,bath_size_Sim_t,bath_size_Sim_v)
Distribution_train,Distribution_val = Sim_train_Input.Sim_Distribution()
train_G_batches = Manipulate.batch_gen(Distribution_train,Sample_Feature_train,X_train.shape[0])
val_G_batches   = Manipulate.batch_gen(Distribution_val,Sample_Feature_val,X_val.shape[0])
#################################################################################
#### Construction de l'Architecture pour Discriminator et Init Parameters
size_input = (Y_train32by32.shape[0],Y_train32by32.shape[1],Y_train32by32.shape[2],Y_train32by32.shape[3])
Discriminator =  discriminator(X_d,size_input,type_nonlinear,type_iniParam)
Discriminator.d_architecture()
output_D = lasagne.layers.get_output(Discriminator.architecture[-1], X_d, deterministic=False)

#### Construction de l'Architecture pour Genarator et Init Parameters
size_dist = (BATCH_SIZE,size_distribution+size_Feature)
Generator =  generative(BATCH_SIZE,size_dist,Z_random)
Generator.g_architecture()
Generate_X = lasagne.layers.get_output(Generator.architecture[-1],Z_random,deterministic=False)
output_G = lasagne.layers.get_output(Discriminator.architecture[-1], Generate_X, deterministic=False)

#################################################################################
#### Le Discrininator sait que echantillon simule est essentiellement simule voir avec 1 coe ds le papier
####  Calcul de la fonction de perte via binary entropy voir papier lecun prevision Video
temp_0G = lasagne.objectives.binary_crossentropy(output_G,T.zeros(BATCH_SIZE))
temp_1G = lasagne.objectives.binary_crossentropy(output_G,T.ones(BATCH_SIZE)).mean()  
temp_G = lasagne.objectives.binary_crossentropy(output_G,T.zeros(BATCH_SIZE)).mean() 
temp_D = lasagne.objectives.binary_crossentropy(output_D,T.ones(BATCH_SIZE)).mean() 
loss_penality = T.mean(lasagne.objectives.squared_error(X_d,X_g))
Peak_Signal_Noise_Ratio = 10*T.log10((T.square(T.max(X_g).astype('float32'))/T.mean(T.square(X_d-X_g).astype('float32'))))
# mean and Peak are the traditionl measure, add structural similarity. take account interaction and and window (look Wikipedia)
loss_G = 0.9*temp_1G + 0.1*(loss_penality)  #Page 4 papier Jan
loss_D = 0.5*temp_D + 0.5*temp_G 
acc_D  = T.mean((T.neq(T.argmax(output_D,axis=1),Y_d)) + (T.neq(T.argmax(output_G,axis=1),Y_g)))
acc_G = T.mean(T.neq(T.argmax(output_G,axis=1),Y_g))  
  # Papier de Lecun page 5 pour expliquer similarite
loss = [loss_D,acc_D,loss_G,acc_G,loss_penality,temp_1G,temp_D,Peak_Signal_Noise_Ratio]
#### Liste des Parametres 
params_D_list = lasagne.layers.get_all_params(Discriminator.architecture,trainable=True)
params_G_list = lasagne.layers.get_all_params(Generator.architecture,trainable=True)
#print(X_d.type(),Generate_X.type(),output_G.type(),output_D.type(),temp_D.type())
#print(params_D_list)
#print(params_G_list)
#################################################################################
Discriminator_params_updates  = lasagne.updates.momentum(loss_D,params_D_list,learning_rate = 0.005,momentum = 0.75)
Generator_params_updates      = lasagne.updates.momentum(loss_G,params_G_list,learning_rate = 0.005,momentum = 0.75)
#Discriminator_params_updates  = lasagne.updates.adam(loss_D,params_D_list,learning_rate = 0.005,beta1 = 0.75)
#Generator_params_updates      = lasagne.updates.adam(loss_G,params_G_list,learning_rate = 0.005,beta1 = 0.75)
#################################################################################
# Fonction Theano pour train and validation
generate_fn_X = theano.function([Z_random],Generate_X)
train_fn_D  = theano.function([X_d,Y_d,X_g,Y_g,Z_random],loss,updates = Discriminator_params_updates,on_unused_input='ignore')
val_fn_D    = theano.function([X_d,Y_d,X_g,Y_g,Z_random],loss,updates = Discriminator_params_updates,on_unused_input='ignore')
train_fn_G   = theano.function([X_d,Y_d,X_g,Y_g,Z_random],loss,updates= Generator_params_updates,on_unused_input='ignore')




for epoch in range(6):
####  Train discriminator with Fixed weight of Generator model
    train_loss_d = 0
    
    train_acc_d = 0
    train_loss_g = 0
    train_acc_g = 0
    nbatch = 0
    for nbatch in range(N_BATCHES): 
        X_d64by64,Y_d32by32,_ = next(train_D_batches)
        Y_dbinary1 = np.ones((BATCH_SIZE)).astype('int32')
        Temp_Noise,Feature,_ = next(train_G_batches)
        sim_distribution = Temp_Noise.eval() 
        Feat_and_Noise = np.c_[sim_distribution,Feature]
        Y_g32by32 = generate_fn_X(Feat_and_Noise)
                
        Y_gbinary0 = np.zeros((BATCH_SIZE)).astype('int32')
        
        Y_g32by32 += 1
        Y_g32by32 *= (255/2)
        Y_g32by32 = np.uint8(Y_g32by32)
        Y_g32by32 = Y_g32by32.astype('float32') 
        
        
        ###########################
        # Update D
        loss = train_fn_D(Y_d32by32,Y_dbinary1,Y_g32by32,Y_gbinary0,Feat_and_Noise)
        train_loss_d += loss[0]
        train_acc_d  += loss[1]
        ###########################
        # Update G
        loss = train_fn_G(Y_d32by32,Y_dbinary1,Y_g32by32,Y_dbinary1,Feat_and_Noise)
        train_loss_g += loss[2]
        train_acc_g  += loss[3]
        ########### 

    print(loss[4],loss[-1])    
    train_loss_d /= N_BATCHES
    train_acc_d /= N_BATCHES
    train_loss_g /= N_BATCHES
    train_acc_g /= N_BATCHES
    ######################################################################################
    val_loss = 0
    val_acc = 0
    nbatch = 0
    for nbatch in range(N_VAL_BATCHES):
        X_d64by64,Y_d32by32,_ = next(val_D_batches)
        Y_dbinary1 = np.ones(BATCH_SIZE).astype('int32')
        Temp_Noise,Feature,_ = next(val_G_batches)
        sim_distribution = Temp_Noise.eval() 
        Feat_and_Noise   = np.c_[sim_distribution,Feature]
        Y_g32by32 = generate_fn_X(Feat_and_Noise)
        Y_gbinary0 = np.zeros(BATCH_SIZE).astype('int32')
        
        Y_g32by32 += 1
        Y_g32by32 *= (255/2)
        Y_g32by32 = np.uint8(Y_g32by32)
        Y_g32by32 = Y_g32by32.astype('float32') 
           
        
        loss = val_fn_D(Y_d32by32,Y_dbinary1,Y_g32by32,Y_gbinary0,Feat_and_Noise)
        val_loss += loss[0]
        val_acc  += loss[1]
                
    val_loss /= N_VAL_BATCHES
    val_acc /= N_VAL_BATCHES
    print('Epoch {}, Train_loss {:.03f}    Valid_Loss {:.03f} ratio {:.03f}'.format(epoch, train_loss_d, val_loss, val_loss/train_loss_d))
    print('         Train_Acc  {:.03f}     Valid_Acc  {:.03f}'.format(100*train_acc_d,100*val_acc))


    
   


#Save


#  np.saazez('disc_params.npz',*[p.get_value() for p in disc_paraaams])
