import os
import numpy as np
import tensorflow as tf
import h5py
import math
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model

from keras.layers import Layer

#moment sets the momentum hyperparameter for the BatchNormalization layers throughout
moment = 0.8


def load_dataset():
    '''
    
    '''
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def identity_block(X, f, filters, batch_norm=True, skip_connection=True, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: python list of integers, defining the number of filters in the CONV layers of the main path
    batch_norm: boolean. If True, apply BatchNormalization layers
    skip_connection: boolean. If True, include skip connections
    initializer: to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X: output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value for the skip connection
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), 
               padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), 
               padding = 'same', kernel_initializer = initializer(seed=0))(X)
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), 
               padding = 'valid', kernel_initializer = initializer(seed=0))(X) 
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    
    if skip_connection:
        # Add shortcut value to main path and pass it through a RELU activation
        X = Add()([X,X_shortcut])
        
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, batch_norm=True, skip_connection=True, s = 2, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters: python list of integers, defining the number of filters in the CONV layers of the main path
    batch_norm: boolean. If True, apply BatchNormalization layers
    skip_connection: boolean. If True, include skip connectionss
    s: Integer, specifying the stride to be used
    initializer: sets up the initial weights of a layer. Defaults to Glorot uniform initializer
    
    Returns:
    X: output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value for the skip connection
    X_shortcut = X

    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), 
               padding='valid', kernel_initializer = initializer(seed=0))(X)
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), 
               padding = 'same', kernel_initializer = initializer(seed=0))(X)
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), 
               padding = 'valid', kernel_initializer = initializer(seed=0))(X) 
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    
    if skip_connection:
        # Shortcut path
        X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s,s), 
                            padding = 'valid', kernel_initializer = initializer(seed=0))(X_shortcut)
        if batch_norm:
            X_shortcut = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
            #X_shortcut = BatchNormalization(axis = 3)(X)
            # Add shortcut value to main path and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        
    X = Activation('relu')(X)
    
    return X
    
    
def resNet_block(X, num_id_blocks, f, filters, s, batch_norm=True, skip_connection=True):
    '''
    Build a ResNet block out of 1 convolutional_block and num_id_blocks identity_blocks
    
    Arguments
        X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        num_id_block: the number of identity_blocks to follow the convolutional_block
        f: the filter size of the middle conv2d layer of each identity_block (integer)
        filters: list containing the sizes of filters to be used in each block
        s: stride length to be used in the convolutional_block
        batch_norm: boolean. If True, apply BatchNormalization layers
        skip_connection: boolean. If True, include skip connections
    Returns
        X: the output of the last identity_block
    '''
    
    X = convolutional_block(X, f = f, filters = filters, batch_norm = batch_norm, skip_connection=skip_connection, s = s)
    
    for _ in range(num_id_blocks):
        X = identity_block(X, f, filters, batch_norm = batch_norm, skip_connection=skip_connection)
        
    return X

def resNet_builder(num_id_blocks_list=[2, 3, 5, 2], f_list=[3, 3, 3, 3], s_list=[1, 2, 2, 2], batch_norm = True,
                   skip_connection=True, first_filters = [64, 64, 256], input_shape = (64, 64, 3), classes = 6):
    '''
    Build a ResNet model with num_ResNet_blocks ResNet blocks.
    Start with a block containing ZeroPadding -> Conv2D -> BatchNorm -> ReLU -> MaxPool.
    The ith ResNet block begins with a convulutional block followed by num_id_blocks_list[i] identity blocks for
    i between 0 and num_id_blocks - 1.
    End with AveragePool -> Flatten -> Dense.
    Default arguments produce a model with ResNet50 architecture
    Arguments
        num_id_blocks_list: list of length num_ResNet_blocks containing the number of id blocks to include 
                            in each ResNet block
        f_list: list of length num_ResNet_blocks of filter sizes to be used in each ResNet block
        s_list: list of length num_ResNet_blocks of stride lengths to be used in each ResNet block
        batch_norm: boolean. If True, apply BatchNormalization layers
        first_filters: list of filter sizes to use in the first ResNet block. Each subsequent block 
                            will double the filter sizes of the previous block.
        input_shape: shape of the images of the dataset
        classes: integer, number of classes
    Returns
        model: a keras Model()
    '''
    
    assert len(num_id_blocks_list) == len(f_list), 'num_id_blocks_list and f_list must be the same length'
    assert len(num_id_blocks_list) == len(s_list), 'num_id_blocks_list and s_list must be the same length'
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    if batch_norm:
        X = BatchNormalization(axis = 3, momentum = moment, epsilon = 1e-6)(X)
        #X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)


    for i in range(len(num_id_blocks_list)):
        filters = [(2**i)*fil for fil in first_filters]
        X = resNet_block(X, num_id_blocks_list[i], f_list[i], filters, s_list[i], batch_norm, skip_connection)  

    # AVGPOOL
    X = AveragePooling2D()(X)
    

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model

def model_tester(model, X_train, Y_train, X_test, Y_test, num_epochs=1, batch_size=32, print_summary=True):
    '''
    Takes in a compiled model, outputs some stats about training time and performance
    Arguments
        model: a compiled model
        X_train, Y_train, X_test, Y_test: training and test data
        num_epochs: numbers of epochs to train the model
        batch_size: batch size to be used in training
        print_summary: Boolean. If True, print a summary of training and evaluation statistics
    returns
        model: trained model
        stats: a dictionary of stats about the trained model
    '''
    import time
    
    stats = {}
    
    tic = time.time()
    model.fit(X_train, Y_train, epochs = num_epochs, batch_size = batch_size)
    toc = time.time()
    
    preds_test = model.evaluate(X_test, Y_test)
    preds_train = model.evaluate(X_train, Y_train)

    stats['Time'] = toc-tic
    stats['Train Loss'] = preds_train[0]
    stats['Train Accuracy'] = preds_train[1]
    stats['Test Loss'] = preds_test[0]
    stats['Test Accuracy'] = preds_test[1]
    
    if print_summary:
        print(f'Time to train for {num_epochs} epochs: {toc-tic:.2f} seconds.')
        print ("Training Loss = " + str(preds_train[0]))
        print ("Training Accuracy = " + str(preds_train[1]))
        print ("Test Loss = " + str(preds_test[0]))
        print ("Test Accuracy = " + str(preds_test[1]))
        
    return model, stats