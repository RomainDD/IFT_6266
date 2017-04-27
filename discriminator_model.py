# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""
#####################################################################
import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import nn
from lasagne.layers.dnn  import Conv2DDNNLayer
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
################################################################################
class discriminator(): 
    def __init__(self,input_var,size_input,type_nonlinear,type_iniParam):
        # Prepare Theano variables for inputs and targets
        self.input_var = input_var  # dim*3*32*32
        self.size_input   = size_input
        self.type_iniParam = type_iniParam
        self.type_nonlinear = type_nonlinear
        #########
        if self.type_nonlinear == 1 :
            self.nonlinear = lasagne.nonlinearities.rectify   
        elif self.type_nonlinear == 2 :
           self.nonlinear = lasagne.nonlinearities.tahn  
        else:
            self.nonlinear = lasagne.nonlinearities.sigmoid 
            
    def fonction_ini(self,mu,sigma,size) :    
        if self.type_iniParam == 1 :
            o_param = lasagne.init.HeNormal(gain ='relu')
        elif self.type_iniParam == 2 :
            o_param = np.random.normal(0.05,1,size)   
        else :
            o_param = np.random.normal(0.05,1,size) 
            
        return o_param   

        
    def d_parameters(self):
        params = {}
        # on passe de 3*32*32 a 96*32*32 
        params["W_D_conv1"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,96,32,32),"W_D_conv1")
        params["b_D_conv1"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(96,),"b_D_conv1")
        #params["W_D_conv1"] = theano.shared(self.fonction_ini(0.05,1,(96,32,32)).astype('float32'))
        #params["b_D_conv1"] = theano.shared(self.fonction_ini(0.0,1,(96)).astype('float32'))
        # on passe de 96*32*32 a 192*32*32   
        params["W_D_conv2"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192,32,32),"W_D_conv2")
        params["b_D_conv2"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(192,),"b_D_conv2")
        # on passe de 192*32*32 a 192
        params["W_D_conv3"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192,32,32),"W_D_conv3")
        params["b_D_conv3"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(192,),"b_D_conv3")
        # on passe de 192 a 192
        params["W_D_conv4"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192),"W_D_conv4")
        params["b_D_conv4"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(1,),"b_D_conv4")
        # on passe de 192 a 2  
        params["W_D_conv5"] = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,2),"W_D_conv5")
        params["b_D_conv5"] = lasagne.utils.create_param(lasagne.init.Constant(0.0),(1,),"b_D_conv5")
        
        return params
    
      
    def d_architecture(self):        
        ### Input
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
        input_shape = self.size_input  # 50000*1*28*28 (channel=1, lenght=28,with=28)
        discriminator_layers = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=self.input_var)]
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],64, (5,5), pad=2,stride=2,W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],128, (5,5), pad=2,stride=2,W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],256, (5,5), pad=2,stride=2,W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],512, (5,5), pad=2,stride=2,W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],1024, (5,5), pad=2,stride=2,W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1], p=0.2))
        discriminator_layers.append(nn.weight_norm(lasagne.layers.NINLayer(discriminator_layers[-1], num_units=1024, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        discriminator_layers.append(lasagne.layers.GlobalPoolLayer(discriminator_layers[-1]))
        #discriminator_layers.append(lasagne.layers.batch_norm(lasagne.layers.DenseLayer(discriminator_layers[-1] , num_units=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.sigmoid)))
        discriminator_layers.append(nn.weight_norm(lasagne.layers.DenseLayer(discriminator_layers[-1] , num_units=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.sigmoid),train_g=True, init_stdv=0.1))
        self.architecture = discriminator_layers
                ##########################
#    
##         ### Input
#        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
#        discriminator_layers = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=self.input_var)]
#        # on passe de 3*32*32  a  96*32*32 car on a 96 filtres 
#        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1], p=0.5))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],64, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 64, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 64, (3,3), pad=1, stride=2, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
#        
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 128, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 128, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 128, (3,3), pad=1, stride=2, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 256, (3,3), pad=0, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 128, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 128, (3,3), pad=1, stride=2, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1], p=0.5))
#        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 512, (3,3), pad=0, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(lasagne.layers.NINLayer(discriminator_layers[-1],num_units=512, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(nn.weight_norm(lasagne.layers.NINLayer(discriminator_layers[-1], num_units=512, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
#        discriminator_layers.append(lasagne.layers.GlobalPoolLayer(discriminator_layers[-1]))       ##########################
#        self.architecture = discriminator_layers                          
#    
                                   
                                  
        
        
        
    
    
    
    def d_architecture2(self):        
        ### Input
        input_shape = self.size_input  # 50000*1*28*28 (channel=1, lenght=28,with=28)
        print(input_shape)
        discriminator_layers = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=self.input_var)]
        # on passe de 3*32*32  a  96*32*32 car on a 96 filtres 
        Tempw = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,96,32,32),"W_D_conv1")
        Tempb =lasagne.utils.create_param(lasagne.init.Constant(0.0),(96,),"b_D_conv1")
        discriminator_layers.append(lasagne.layers.Conv2DLayer(discriminator_layers[-1],num_filters=96,filter_size=(5, 5),nonlinearity=self.nonlinear,W=Tempw,b=Tempb))
        # check if convolution type (full, same,valid)
        #########################
        # on passe de 96*32*32 a 96*16*16  car pooling 2*2 
        #discriminator_layers.append(lasagne.layers.MaxPool2DLayer(discriminator_layers[-1], pool_size=(2,2)))
        # on passe de 96*32*32 a 192*32*32  
        Tempw = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192,32,32),"W_D_conv2")
        Tempb =lasagne.utils.create_param(lasagne.init.Constant(0.0),(192,),"b_D_conv2")
        #Tempw = theano.shared(self.fonction_ini(0.05,1,(1,192,32,32)).astype('float32'))
        #Tempb = theano.shared(self.fonction_ini(0.05,1,(192,)).astype('float32'))
        discriminator_layers.append(lasagne.layers.Conv2DLayer(discriminator_layers[-1],num_filters=192,filter_size=(5, 5),nonlinearity=self.nonlinear,W=Tempw,b=Tempb))
         # on passe de 192*32*32 a 192
        Tempw = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192,32,32),"W_D_conv3")
        Tempb =lasagne.utils.create_param(lasagne.init.Constant(0.0),(192,),"b_D_conv3")
        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1],p=.5))
        discriminator_layers.append(lasagne.layers.Conv2DLayer(discriminator_layers[-1],num_filters=192,filter_size=(5, 5),nonlinearity=self.nonlinear,W=Tempw,b=Tempb))
        # on passe de 192 a 192
        Tempw = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,192),"W_D_conv4")
        Tempb =lasagne.utils.create_param(lasagne.init.Constant(0.0),(1,),"b_D_conv4")
        discriminator_layers.append(lasagne.layers.NINLayer(discriminator_layers[-1],num_units = 192,nonlinearity=self.nonlinear,W=Tempw,b=Tempb))
        discriminator_layers.append(lasagne.layers.GlobalPoolLayer(discriminator_layers[-1]))
        # on passe de 92 a 2 
        Tempw = lasagne.utils.create_param(lasagne.init.Normal(0.05),(1,2),"W_D_conv5")
        Tempb =lasagne.utils.create_param(lasagne.init.Constant(0.0),(1,),"b_D_conv5")
        discriminator_layers.append(lasagne.layers.DenseLayer(discriminator_layers[-1], num_units=2, W=Tempw,b=Tempb,nonlinearity=None))
        print(input_shape)
        self.parameters = lasagne.layers.get_all_params(discriminator_layers, trainable=True)
        self.last_layer = discriminator_layers[-1]
        ##########################
        self.architecture = discriminator_layers
        
        
        input_shape = self.size_input  # 50000*1*28*28 (channel=1, lenght=28,with=28)
        
        discriminator_layers = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=self.input_var)]
        # on passe de 3*32*32  a  96*32*32 car on a 96 filtres 
        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1], p=0.5))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1],192, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 192, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 192, (3,3), pad=1, stride=2, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.Leakyrectify)))
        
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 256, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 256, (3,3), pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 256, (3,3), pad=1, stride=2, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(lasagne.layers.DropoutLayer(discriminator_layers[-1], p=0.5))
       
        discriminator_layers.append(nn.weight_norm(Conv2DDNNLayer(discriminator_layers[-1], 256, (3,3), pad=0, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(nn.weight_norm(lasagne.layers.NINLayer(discriminator_layers[-1],num_units=256, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(nn.weight_norm(lasagne.layers.NINLayer(discriminator_layers[-1], num_units=256, W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.rectify)))
        discriminator_layers.append(lasagne.layers.GlobalPoolLayer(discriminator_layers[-1]))

        self.architecture = disc_layers
        
        
        return
        
    def d_loss_function(self,X_true,Y_true): 
        ###  may be reshape if not
         #Y = T.ones(self.prediction.shape)  # Papier Deep Multiscale video
        ### on applique reseau D avec Input issue du training set 
        self.prediction_X_true = lasagne.layers.get_output(self.last_layer,X_true,deterministic=False)
        #self.prediction_X_generate = lasagne.layers.get_output(self.last_layer,X_generate,deterministic=False)
        loss_d = Jan.log_sum_exp(self.prediction_X_true)
        #loss_d = lasagne.objectives.binary_crossentropy(self.prediction_X_true,Y_true)
        loss_d = lasagne.objectives.aggregate(loss_d, mode='mean')
        acc_d = T.mean(T.eq(T.argmax(self.prediction_X_true, axis=1),Y_true),dtype=theano.config.floatX)
        self.params_d = lasagne.layers.get_all_params(self.last_layer,trainable = True)
        #updates = lasagne.updates.adadelta(loss_train,all_params)
        updates = lasagne.updates.nesterov_momentum(loss_d,self.params_d,learning_rate=0.01,momentum=0.9)
        ########
        return  loss_d, acc_d, updates 
        
        
        
  
