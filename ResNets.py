'''
This code is based on an assignment from the DeepLearning.AI's Coursera course titled Convolutional Neural 
Networks. I have expanded on the assignment by adding functions to facilitate model evaluation and 
generalizing the architecture to include any number of layers with or without skip connections and with or 
without batch normalization.

Skills demonstrated: multi-classification, tensorflow, ResNet architecture, model evaluation, general coding

Overview: 
    ResNet architecture (He et al, 2015) famously revolutionized deep learning by elegantly addressing the
    issue of vanishing gradients in very deep networks by adding skip connections. This code builds a deep
    neural network using ResNet architecture. For the sake of comparison with pre-ResNet architectures, I
    include helper functions that easily allow the removal of skip connections. The model is trained on images
    of hands holding up 0-5 fingers and returns an integer corresponding to the gesture in each image.

Model(s):
    - Images are initially fed into an input layer, then through a ZeroPadding layer, a Conv2D layer, 
      BatchNormalization, ReLU activation, then a MaxPool layer. From there, output is passed through a
      prescribed number of what I'll call ResNet blocks. Each ResNet block contains a convolutional block 
      and a prescribed number of identity blocks.
    - The convolutional blocks pass the input through three Conv2D layers with BatchNormalization and ReLU
      activation between, then pass the block input through a Conv2D and BatchNorm and adds it to the output of
      the third Conv2D layer before passing that sum through a ReLU activation. These blocks can change the
      dimension of the data.
    - The identity blocks similarly pass the input through three Conv2D layers with BatchNormalization and
      ReLU activation between, but simply adds the output of the third Conv2D layer to the block input without
      transforming the input first. These blocks maintain data dimensions.
    - After the last ResNet block, the output is passed through an AveragePooling2D layer, then a Flatten layer,
      then a Dense layer with softmax activation for prediction.
    * BatchNormalization and skip connections can be removed using the corresponding boolean arguments in the
      resNet_builder function

Purpose: 
- Demonstrate the effectiveness of skip connections in addressing the issue of vanishing gradients in very 
  deep neural networks.

Datasets: 
- SIGNS dataset from DeepLearning.AI's Convolutional Neural Networks course. The dataset consists of 1080 
  training images and 120 test images of hands gesturing 0 through 5.
    

Target variable:
- The number of digits held up in each image (0-5).
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
from tensorflow.keras.utils import to_categorical

from resnets_utils import *

np.random.seed(1)
tf.random.set_seed(2)

#model_b = resNet_builder([2, 3, 2], [3, 3, 3], [1, 2, 2], batch_norm = True, skip_connection=True,
#                   first_filters = [64, 64, 256], input_shape = (64, 64, 3), classes = 6)
model_b = resNet_builder()

#print(model_b.summary())

opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
model_b.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = np.squeeze(to_categorical(Y_train_orig, num_classes=6))
Y_test = np.squeeze(to_categorical(Y_test_orig, num_classes=6))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model_t, stats = model_tester(model_b, X_train, Y_train, X_test, Y_test, num_epochs=5, batch_size=32)
