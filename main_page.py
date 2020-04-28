# Tensor version is 2.0.0
from writing import Writelogs
from Loadrawdata_show_2 import Loadrawdata_show
from Normalization_raw_data_3 import NoramlPic
from CNN_layer import CNN_model
from Create_model import Create_model
from Compile_model import Compilemodel
from Train_model import Trainmodel
from Evaluate_model import Evaluate_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#from mini_bach import random_mini_batches
#load raw data from files and show to user
loadrawdata=Loadrawdata_show()
loadrawdata.load
#index of the picture that user wnats to see
#loadrawdata[3]
mywriting = Writelogs()
mywriting.writing(str(loadrawdata))
#print(loadrawdata)
###########################################################
#Normalization of raw data
norm_data=NoramlPic()
norm_data.norm(loadrawdata.load)
mywriting.writing(str(norm_data))
#print(norm_data)
#####################################################
# create model
# input_shape, ziropad, no_filter, conv_filter_size, conv_stride, conv_activ_func, pool_filter_size,fully_activ_func, modelname
cnn=CNN_model()#return X
# input_shape, ziropad, no_filter, conv_filter_size, conv_stride, conv_activ_func, pool_filter_size
cnn.cnn_layer(norm_data.X_train.shape[1:], 3, 32, 7, 1, 'relu', 2)#return X
creating_model=Create_model()#return model
creating_model.create_model(cnn.X,norm_data.len_class,'softmax', 'fc', cnn.X_input, "fingerModel")#return model
compile_model=Compilemodel()
compile_model.compilemodel(creating_model.model_)
train_model=Trainmodel()
train_model.trainmodel(norm_data.X_train,norm_data.Y_train, creating_model.model_,epoch=1, batch_size=64)
evaluate_model=Evaluate_model()

evaluate_model.evaluatemodel(norm_data.X_test,norm_data.Y_test,creating_model.model_)