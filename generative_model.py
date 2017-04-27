# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:39:52 2017

@author: David
"""
#####################################################################
import theano
import lasagne
import lasagne.layers as ll
import numpy as np
import theano.tensor as T
from lasagne import layers,init
from theano.sandbox.rng_mrg import MRG_RandomStreams
#######
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer,TransposedConv2DLayer,DenseLayer,ReshapeLayer,batch_norm, Deconv2DLayer
    print('cuda available')
except ImportError:
    from lasagne.layers import Conv2DLayer, MaxPool2DLayer,TransposedConv2DLayer,DenseLayer,ReshapeLayer,batch_norm, Deconv2DLayer
########
try:
   import cPickle as pickle
except:
   import pickle

theano.config.floatX = 'float32'
import nn


###############################################################################
X_true = T.tensor4('inputs')
X_generate = T.tensor4('inputs_generate')
Y_true = T.ivector('targets')   
Y_generate = T.ivector('targets') 

###############################################################################
#################################################################################
class generative():  
    
    def __init__(self,BATCH_SIZE,size_distribution,Z):
        
        self.BATCH_SIZE = BATCH_SIZE
        self.size_input = size_distribution
        self.distribution = Z
         
    def fonction_ini(self,size) :    
        if self.type_iniParam == 1 :
            o_param = lasagne.init.HeNormal(gain ='relu')
        elif self.type_iniParam == 2 :
            o_param = np.random.normal(0.05,1,size)   
        else :
            o_param = np.random.normal(0.05,1,size) 
        return o_param   


    def g_ini_parameters(self): 
            ####################################  papier de DCGAN
        nbre_chan = 3            # RBG Channel 
        gifn = Radfor.Normal(scale=0.02)
        difn = Radfor.Normal(scale=0.02)
        gain_ifn = Radfor.Normal(loc=1., scale=0.02)
        bias_ifn = Radfor.Constant(c=0.)
        nbre_filtre_G = self.nbre_filtre  # Nbre de filtre  
        
        gw  = gifn((self.size_distribution, nbre_filtre_G*8*4*4), 'gw')
        gg = gain_ifn((nbre_filtre_G*8*4*4), 'gg')
        gb = bias_ifn((nbre_filtre_G*8*4*4), 'gb')
        gw2 = gifn((nbre_filtre_G*8, nbre_filtre_G*4, 5, 5), 'gw2')
        gg2 = gain_ifn((nbre_filtre_G*4), 'gg2')
        gb2 = bias_ifn((nbre_filtre_G*4), 'gb2')
        gw3 = gifn((nbre_filtre_G*4,nbre_filtre_G*2, 5, 5), 'gw3')
        gg3 = gain_ifn((nbre_filtre_G*2), 'gg3')
        gb3 = bias_ifn((nbre_filtre_G*2), 'gb3')
        gw4 = gifn((nbre_filtre_G*2, nbre_filtre_G, 5, 5), 'gw4')
        gg4 = gain_ifn((nbre_filtre_G), 'gg4')
        gb4 = bias_ifn((nbre_filtre_G), 'gb4')
        gwx = gifn((nbre_filtre_G,nbre_chan, 5, 5), 'gwx')
        Generator_ini_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
        return Generator_ini_params

    def g_ini_parameters2(self):
        params = {}
        # on passe de 100 a 8192=4*4*512
        params["W_G_dconv1"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,512*4*4),"W_G_dconv1")
        #params["b_G_dconv1"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(nbre_filtre*8*4*4,),"b_G_dconv1")
        # params["W_G_dconv1"] = theano.shared(self.fonction_ini(0.05,1,(4*4*512)).astype('float32'))
        # params["b_G_dconv1"] = theano.shared(self.fonction_ini(0.0,(1)).astype('float32'))
        # on passe de 8192 a 256*8*8 
        params["W_G_dconv2"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,256,8,8),"W_G_dconv2")
        #params["W_G_dconv2"] = theano.shared(self.fonction_ini(0.05,1,(128,8,8)).astype('float32'))
        #params["b_G_dconv2"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(128,),"b_G_dconv2")
        # on passe de 256*8*8 a 128*16*16
        
        params["W_G_dconv3"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,128,16,16),"W_G_dconv3")
        #params["b_G_dconv3"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(128,),"b_G_dconv3")
        # on passe de 128*16*16 a 3*32*32
        params["W_G_dconv4"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,3,32,32),"W_G_dconv4")
        #params["b_G_dconv4"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(3,),"b_G_dconv4")
        #gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]
        return params
    
   
        
    def g_architecture2(self,params):
         
        ### Input
        gen_layers = [lasagne.layers.InputLayer(shape=self.size_input,input_var=self.distribution)]
        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_layers[-1], num_units=4*4*512, W=params["W_G_dconv1"], nonlinearity=nn.relu)))
        gen_layers.append(lasagne.layers.ReshapeLayer(gen_layers[-1], (self.size_input[0],512,4,4)))
        gen_layers.append(lasagne.layers.batch_norm(Jan.Deconv2DLayer(gen_layers[-1], (self.size_input[0],256,8,8), (3,3), W=params["W_G_dconv2"], nonlinearity=nn.relu))) # 4 -> 8
        gen_layers.append(lasagne.layers.batch_norm(Jan.Deconv2DLayer(gen_layers[-1], (self.size_input[0],128,16,16),(3,3), W=params["W_G_dconv3"], nonlinearity=nn.relu))) # 8 -> 16
        gen_layers.append(Jan.weight_norm(Jan.Deconv2DLayer(gen_layers[-1],(self.size_input[0],3,32,32), (3,3), W=params["W_G_dconv4"], nonlinearity=nn.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
        #gen_dat = ll.get_output(gen_layers[-1])
        self.architecture = gen_layers
        self.last_layer   = gen_layers[-1]
        self.parameters = lasagne.layers.get_all_params(gen_layers,trainable=True)
        return 

    
    
    def g_architecture(self):
#        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
#        gen_layers = [lasagne.layers.InputLayer(shape=self.size_input, input_var=self.distribution)]
#        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_layers[-1], num_units=4*4*512, nonlinearity=lrelu)))
#        gen_layers.append(lasagne.layers.ReshapeLayer(gen_layers[-1], (-1,512,4,4)))
#        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=256, filter_size=5, stride=2, crop=2,output_size=8,nonlinearity=lasagne.nonlinearities.rectify)))
#        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=128, filter_size=5, stride=2, crop=2,output_size=16,nonlinearity=lasagne.nonlinearities.rectify)))
#        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=64, filter_size=5, stride=2, crop=2,output_size=32,nonlinearity=lasagne.nonlinearities.rectify)))
#        #gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=3, filter_size=5, stride=2, crop=2,output_size=64,W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.tanh))) # 16 -> 32
#        gen_layers.append(nn.weight_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=3, filter_size=5, stride=2, crop=2,output_size=64,W=lasagne.init.Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) 
#        self.architecture = gen_layers
###        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(gen_layers[-1], num_filters=3, filter_size=5, stride=2, crop=2,output_size=32,nonlinearity=lasagne.nonlinearities.sigmoid)))
#
#        self.architecture = gen_layers
#        #self.last_layer   = gen_layers[-1]
        #self.parameters = lasagne.layers.get_all_params(gen_layers,trainable=True)
#
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
        gen_layers = [lasagne.layers.InputLayer(shape=self.size_input, input_var=self.distribution)]
        #gen_layers.append(lasagne.layers.DropoutLayer(gen_layers[-1], p=0.5))
        gen_layers.append(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_layers[-1], num_units=4*4*512, nonlinearity=lasagne.nonlinearities.rectify)))
        gen_layers.append(lasagne.layers.ReshapeLayer(gen_layers[-1], (self.BATCH_SIZE,512,4,4)))
        gen_layers.append(lasagne.layers.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.BATCH_SIZE,256,8,8), (5,5), W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))# 4 -> 8
        gen_layers.append(lasagne.layers.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.BATCH_SIZE,128,16,16), (5,5), W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify))) # 4 -> 8
        gen_layers.append(lasagne.layers.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (self.BATCH_SIZE,64,32,32), (5,5), W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify))) # 4 -> 8
        #gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (self.BATCH_SIZE,3,64,64), (5,5), W=init.Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1))  
        gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (self.BATCH_SIZE,3,64,64), (5,5), W=init.Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1))  # 16 -> 32, train_g=True, init_stdv=0.1
        self.architecture = gen_layers
   



        return       
       
        
    def g_loss_function(self,X_generate,Y_generate,Last_layer): 
        ###  may be reshape if not  X and output_generate
        ### on applique reseau D avec Input issue du model generate
        ###X_generate = lasagne.layers.get_output(self.architecture[-1])   # 3*32*32 du Generative Model
        self.prediction_X_generate = lasagne.layers.get_output(Last_layer,X_generate,deterministic=False)
        #Y = T.ones(self.prediction.shape)  # Papier Deep Multiscale video
        #loss_g = lasagne.objectives.binary_crossentropy(self.prediction_X_generate,Y_generate)
        loss_g = Jan.log_sum_exp(self.prediction_X_generate)
        loss_g = lasagne.objectives.aggregate(loss_g, mode='mean')
        #loss_penality = T.sum(lasagne.objectives.squared_error(X_true,X_generate))
        #lloss = 0.5*loss_g  + 0.5*loss_penality  # voir si hyperparametre
        loss = loss_g
        acc = T.mean(T.eq(T.argmax(self.prediction_X_generate,axis=1),Y_generate),dtype=theano.config.floatX)
        params_g = lasagne.layers.get_all_params(self.last_layer,trainable = True)
        #updates = lasagne.updates.adadelta(loss,params_g)  # Recomandation video GAN tutorial
        updates = lasagne.updates.nesterov_momentum(loss,params_g,learning_rate=0.01,momentum=0.9)
        # A completer Training
        #self.train_fn = theano.function([X_true,X_generate,Y],loss,updates=updates)
        #self.val_fn = theano.function([X_true,X_generate,Y],loss])
        #pred_fn = theano.function([X_true], prediction)    
        return loss, acc, updates
       