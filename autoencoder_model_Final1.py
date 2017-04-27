import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import pylab  
import nn
import matplotlib.pyplot as plt 
from lasagne.layers.dnn  import Conv2DDNNLayer
import skimage.measure
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
from manipulate_Data_Benchmark_Final1 import manipulate   
Manipulate = manipulate(32,32,10)


################################################################################
class autoencoder(): 
    def __init__(self,Size_Input,Size_Encode,Size_Epoch,Batch_Size,Size_Filter):
        # Prepare Theano variables for inputs and targets
        self.Size_Encode = Size_Encode  # Size Encodeur
        self.Size_Input = Size_Input
        self.Size_Epoch = Size_Epoch
        self.Batch_Size = Batch_Size
        self.Size_Filter = Size_Filter
        
       
                
    def build_network(self,X,Y):
        # Define the layers 
        
        
        lrelu = lasagne.nonlinearities.LeakyRectify(0.1)
        input_shape = self.Size_Input 
        #temp = input_shape[1]*input_shape[2]*input_shape[3]
        Auto_Enc_Layer = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=X)]
        # Encode lasagne.nonlinearities.rectify
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],64,(3,3),pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-2],64,(3,3),pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        
        Auto_Enc_Layer_Global0 = nn.weight_norm(lasagne.layers.NINLayer(Auto_Enc_Layer[-1], num_units=32, W=lasagne.init.Normal(0.05), nonlinearity=lrelu))
        Auto_Enc_Layer_Global = lasagne.layers.GlobalPoolLayer(Auto_Enc_Layer_Global0)

        
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],64,(3,3),pad=1,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        Auto_Enc_Layer.append(lasagne.layers.DropoutLayer(Auto_Enc_Layer[-1], p=0.5))
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],128,(3,3),pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],128,(3,3),pad=1, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        Auto_Enc_Layer.append(nn.weight_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],128,(3,3),pad=1,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
                
        Auto_Enc_Layer_Local0 = nn.weight_norm(lasagne.layers.NINLayer(Auto_Enc_Layer[-1], num_units=128, W=lasagne.init.Normal(0.05), nonlinearity=lrelu))
        Auto_Enc_Layer_Local = lasagne.layers.GlobalPoolLayer(Auto_Enc_Layer_Local0)
       
        Auto_Enc_Layer.append(nn.weight_norm(lasagne.layers.NINLayer(Auto_Enc_Layer[-1], num_units=512, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
        Auto_Enc_Layer.append(lasagne.layers.GlobalPoolLayer(Auto_Enc_Layer[-1]))
       
        Auto_Enc_Layer.append(nn.weight_norm(DenseLayer(Auto_Enc_Layer_Local,num_units=32*16*16,W=lasagne.init.Normal(0.05),nonlinearity=lasagne.nonlinearities.tanh)))
        # Decccode
      
        Auto_Dec_Layer=[(lasagne.layers.ReshapeLayer(Auto_Enc_Layer[-1], (self.Batch_Size,32,16,16)))]
        Auto_Dec_Layer.append(nn.weight_norm(nn.Deconv2DLayer(Auto_Dec_Layer[-1], (self.Batch_Size,32,32,32), (3,3), W=lasagne.init.Normal(0.05), nonlinearity=lrelu))) # 4 -> 8
        Auto_Dec_Layer.append(nn.weight_norm(nn.Deconv2DLayer(Auto_Dec_Layer[-1], (self.Batch_Size,3,64,64), (3,3), W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.tanh))) # 4 -> 8
         
        all_params = lasagne.layers.get_all_params(Auto_Dec_Layer,trainable=True)
        network_output = lasagne.layers.get_output(Auto_Dec_Layer[-1], X, deterministic=False)
        
        
        
        encoded_output  = lasagne.layers.get_output(Auto_Enc_Layer_Local, X, deterministic=True)
        encoded_output1 = lasagne.layers.get_output(Auto_Enc_Layer_Global, X, deterministic=True)
        #encoded_output1 = Auto_Enc_Layer_Global 
        network_output1 = lasagne.layers.get_output(Auto_Dec_Layer[-1], X, deterministic=True)
        
        loss_A = T.mean(lasagne.objectives.squared_error(network_output[:,0,:,:],Y[:,0,:,:])) + T.mean(lasagne.objectives.squared_error(network_output[:,1,:,:],Y[:,1,:,:])) + T.mean(lasagne.objectives.squared_error(network_output[:,2,:,:],Y[:,2,:,:]))
      
        #loss_A = T.mean(lasagne.objectives.squared_error(network_output,Y))
      
             
        loss = [loss_A,encoded_output,network_output]
        #Autoencodeur_params_updates = lasagne.updates.momentum(loss_A,all_params,learning_rate = 0.05,momentum = 0.5)
        Autoencodeur_params_updates  = lasagne.updates.adam(loss_A,all_params,learning_rate = 0.001,beta1 = 0.9)
        # Some Theano functions ,
        
        
        self.generate_fn_X = theano.function([X],network_output)
        self.train = theano.function([X,Y],loss,updates=Autoencodeur_params_updates,allow_input_downcast=True)
        self.predict = theano.function([X],network_output1,allow_input_downcast=True)
        self.encode_L = theano.function([X],encoded_output,allow_input_downcast=True)
        self.encode_G = theano.function([X],encoded_output1,allow_input_downcast=True)
        #s#elf.encode_G =  encoded_output1
        self.network = Auto_Enc_Layer
        
        
        
        return 
    
    
      
    def batch_gen(self,X,Y,size):
        while True:
            idx = np.random.choice(size,self.Batch_Size)
            yield X[idx].astype('float32'),Y[idx].astype('float32')
     
   
    def learn(self,X,Y):
        Image_Test_PR = Y[0:32,:,:,:].astype('float32')
        X_d = T.tensor4()
        X_g = T.tensor4()
        temp = X.shape[0]
        Peak_Signal_Noise_Ratio = 10*T.log10((T.square(T.max(X_g).astype('float32'))/T.mean(T.square(X_d-X_g).astype('float32'))))
        #Peak_Signal_Noise_Ratio =  skimage.measure.compare_ssim(X_d.astype('float32'),X_g.astype('float32'))
        Peak_Signal_Noise_Ratio_fn = theano.function([X_d,X_g],Peak_Signal_Noise_Ratio)
        train_batches = self.batch_gen(X,Y,temp)
        N_BATCHES = temp // self.Batch_Size
        Epoch_train_loss_AE = []
        Epoch_train_SN_Ratio_AE = []
        
        for nepoch in range(self.Size_Epoch):
            train_loss = 0 
            nbatch = 0
            train_loss_AE= []
            train_acc_AE = []
            for nbatch in range(self.Batch_Size):     
                X_b,Y_b = next(train_batches)
                
                Rec_Y_b = self.generate_fn_X(X_b)
                Y_b = Y_b/(255/2)  
                Y_b = Y_b-1
                
                temp = self.train(Rec_Y_b,Y_b)
                train_loss += temp[0]
                
                
            Rec_Y_b_Test = self.predict(Image_Test_PR[0:32,:,:,:])
            Temp = Rec_Y_b_Test + 1
            Temp = Temp*(255/2)   
            P_Signal_Noise_R6 = np.float(Peak_Signal_Noise_Ratio_fn(Image_Test_PR[0:1,:,:,:],Temp[0:1,:,:,:]))
            P_Signal_Noise_R7 = np.float(Peak_Signal_Noise_Ratio_fn(Image_Test_PR[1:2,:,:,:],Temp[1:2,:,:,:]))  
            P_Signal_Noise_R8 = np.float(Peak_Signal_Noise_Ratio_fn(Image_Test_PR[31:32,:,:,:],Temp[31:32,:,:,:]))

           # P_Signal_Noise_R9,g1,g2 = skimage.measure.compare_ssim(Image_Test_PR[0,1,:,:],Temp[0,1,:,:])
#            P_Signal_Noise_R7 = skimage.measure.compare_ssimImage_Test_PR[1,:,:,:],Temp[1,:,:,:])
#            P_Signal_Noise_R8 = skimage.measure.compare_ssim(Image_Test_PR[2,:,:,:],Temp[2,:,:,:])
            
            Temp0 = (P_Signal_Noise_R6+ P_Signal_Noise_R7 + P_Signal_Noise_R7)/3
                   
            train_loss /= N_BATCHES
            Epoch_train_loss_AE.append(train_loss) 
            Epoch_train_SN_Ratio_AE.append(Temp0)
           
            print(P_Signal_Noise_R6,P_Signal_Noise_R7,P_Signal_Noise_R8,Temp0)
            print("Epoch {} average loss = {}".format(nepoch, train_loss)) 
          
        return  Epoch_train_loss_AE, Epoch_train_SN_Ratio_AE      
            
            
#            T1=2
#            T2=2
#            plt.figure(figsize=(T2,T1))
#            Manipulate.plot_img(Image_Test_PR[0,:,:,:],Temp[0,:,:,:],0,T1,T2)
#            Manipulate.plot_img(Image_Test_PR[1,:,:,:],Temp[1,:,:,:],1,T1,T2)
            
            #print(Y_b.max(),Y_b.min(),Rec_Y_b.max(),Rec_Y_b.min())
         # en effet sur le sample de Test,le nouveau X_train est encoder avec les parametres du modele retenu 
        
        
#if __name__ == "__main__":
#   trainfunc, predict, encode = self.build_network()
#   self.learn()
#   #self.check_model(x_test, predict, encode)
#
# lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
#        input_shape = self.Size_Input 
#        #temp = input_shape[1]*input_shape[2]*input_shape[3]
#        Auto_Enc_Layer = [lasagne.layers.InputLayer(shape=(None,input_shape[1],input_shape[2],input_shape[3]),input_var=X)]
#        # Encode lasagne.nonlinearities.rectify
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],256,(9,9),pad=2,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(DenseLayer(Auto_Enc_Layer[-1],num_units=32,W=lasagne.init.Normal(0.05),nonlinearity=lrelu)))
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(Conv2DDNNLayer(Auto_Enc_Layer[-2],256,(7,7),pad=2,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
#        
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],256,(5,5),pad=2,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(Conv2DDNNLayer(Auto_Enc_Layer[-1],256,(3,3),pad=2,stride =2, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
#     
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(DenseLayer(Auto_Enc_Layer[-1],num_units=128,W=lasagne.init.Normal(0.05),nonlinearity=lrelu)))
##        Auto_Enc_Layer.append(lasagne.layers.batch_norm(lasagne.layers.NINLayer(Auto_Enc_Layer[-1], num_units=128, W=lasagne.init.Normal(0.05), nonlinearity=lrelu)))
##        Auto_Enc_Layer.append(lasagne.layers.GlobalPoolLayer(Auto_Enc_Layer[-1]))
#        Auto_Enc_Layer.append(lasagne.layers.batch_norm(DenseLayer(Auto_Enc_Layer[-1],num_units=32*16*16,W=lasagne.init.Normal(0.05),nonlinearity=lasagne.nonlinearities.tanh)))
#        # Decccode
#      
#        Auto_Dec_Layer=[(lasagne.layers.ReshapeLayer(Auto_Enc_Layer[-1], (self.Batch_Size,32,16,16)))]
#        Auto_Dec_Layer.append(lasagne.layers.batch_norm(nn.Deconv2DLayer(Auto_Dec_Layer[-1], (self.Batch_Size,32,32,32), (3,3), W=lasagne.init.Normal(0.05), nonlinearity=lrelu))) # 4 -> 8
#        Auto_Dec_Layer.append(lasagne.layers.batch_norm(nn.Deconv2DLayer(Auto_Dec_Layer[-1], (self.Batch_Size,3,64,64), (3,3), W=lasagne.init.Normal(0.05), nonlinearity=lasagne.nonlinearities.tanh))) # 4 -> 8
#         
#        all_params = lasagne.layers.get_all_params(Auto_Dec_Layer,trainable=True)
#        network_output = lasagne.layers.get_output(Auto_Dec_Layer[-1], X, deterministic=False)
#        
#    
    
    