from writing import Writelogs
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
#import pydot
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
#from kt_utils import *
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
class Create_model():
    def create_model(self,X,no_class,fully_activ_func,fully_name,X_input_,modelname):

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(no_class, activation=fully_activ_func, name='fc')(X)

        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        self.model_ = Model(inputs=X_input_, outputs=X, name=modelname)
        print("-----initialzing model is done----")
        return self.model_.name




