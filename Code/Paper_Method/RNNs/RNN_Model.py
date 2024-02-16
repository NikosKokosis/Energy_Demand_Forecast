#!/usr/bin/env python
# coding: utf-8

# In[13]:


#get_ipython().system('jupyter nbconvert --to script RNN_Model.ipynb')


# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Bidirectional
from tensorflow.keras.regularizers import l2


# In[15]:


def RnnModel(model_type, input_shape, num_layers, num_units, regularization):
    """
    Create a recurrent neural network model.

    Parameters:
        model_type (str): Type of RNN model to use. Options: 'LSTM', 'SimpleRNN', 'Bi-LSTM' or 'GRU'.
        input_shape (tuple): Shape of input data (sequence_length, num_features).
        num_layers (int): Number of recurrent layers to stack.
        num_units (int): Number of units (neurons) in each recurrent layer.
        regularization (float): Strength of L2 regularization.

    Returns:
        model: A Keras Model object representing the created RNN model.
    """
    # Initialize a Sequential model
    model = Sequential()
    
    # Loop through the specified number of layers
    for i in range(num_layers):
        if model_type == 'LSTM':
            # Add LSTM layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(LSTM(units=num_units, activation='tanh', use_bias=True,
                               recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                               return_sequences=True, input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(LSTM(units=num_units, activation='tanh', use_bias=True, 
                               recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                               return_sequences=False, input_shape=input_shape))

        elif model_type == 'GRU':
            # Add GRU layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(GRU(units=num_units, activation='tanh', use_bias=True,
                              recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                              return_sequences=True, input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(GRU(units=num_units, activation='tanh', use_bias=True, 
                               recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                               return_sequences=False, input_shape=input_shape))

        elif model_type == 'Bi-LSTM':
            # Add Bidirectional LSTM layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(Bidirectional(LSTM(units=num_units, activation='tanh', use_bias=True,
                                             recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                                             return_sequences=True), input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(Bidirectional(LSTM(units=num_units, activation='tanh', use_bias=True, 
                               recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                               return_sequences=False, input_shape=input_shape)))

        elif model_type == 'SimpleRNN':
            # Add SimpleRNN layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(SimpleRNN(units=num_units, activation='tanh', use_bias=True,
                                    recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                                    return_sequences=True, input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(SimpleRNN(units=num_units, activation='tanh', use_bias=True, 
                               recurrent_regularizer=l2(regularization), kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization),
                               return_sequences=False, input_shape=input_shape))

        else:
            # Raise an error for invalid model type
            raise ValueError("Invalid model type. Please choose from 'LSTM', 'SimpleRNN', 'Bi-LSTM' or 'GRU'.")
        
    # Add output Dense layer
    model.add(Dense(units=1))
    
    return model


# In[16]:


def train_rnn_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size):
    """
    Train an RNN model.
    
    Args:
    - model: The RNN model to train.
    - X_train: Input data for training.
    - Y_train: Target data for training.
    - X_val: Input data for validation.
    - Y_val: Target data for validation.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    
    Returns:
    - History object containing training/validation loss and metric values.
    """
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val)
    )
    return history

