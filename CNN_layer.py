from writing import Writelogs
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
# from kt_utils import *
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
class CNN_model():
    def cnn_layer(self, input_shape, ziropad, no_filter, conv_filter_size, conv_stride, conv_activ_func,
                 pool_filter_size):
        """
        Implementation of the HappyModel.

        Arguments:
        input_shape -- shape of the images of the dataset
            (height, width, channels) as a tuple.
            Note that this does not include the 'batch' as a dimension.
            If you have a batch like 'X_train',
            then you can provide the input_shape using
            X_train.shape[1:]
        """
        ###Returns:
        # model -- a Model() instance in Keras
        """
        
        ### START CODE HERE ###
        # Feel free to use the suggested outline in the text above to get started, and run through the whole
        # exercise (including the later portions of this notebook) once. The come back also try out other
        # network architectures as well. 
        """
        self.X_input = Input(input_shape)
        self.X = ZeroPadding2D((ziropad, ziropad))(self.X_input)
        # CONV -> BN -> RELU Block applied to X
        self.X = Conv2D(no_filter, (conv_filter_size, conv_filter_size),
                        strides=(conv_stride, conv_stride), name='conv0')(self.X)
        self.X = BatchNormalization(axis=3, name='bn0')(self.X)
        self.X = Activation(conv_activ_func)(self.X)
        # MAXPOOL
        self.X = MaxPooling2D((pool_filter_size, pool_filter_size), name='max_pool')(self.X)
        return self.X,self.X_input
