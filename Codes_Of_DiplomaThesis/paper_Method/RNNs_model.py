#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Bidirectional, Dropout
from tensorflow.keras.regularizers import l2


# In[ ]:


def RnnModel(model_type, input_shape, num_layers, num_units, recurrent_activation_function,
             recurrent_regularization, kernel_regularization, bias_regularization,
             dropout_rate, recurrent_dropout_rate):
    """
    Create a recurrent neural network model.

    Parameters:
        model_type (str): Type of RNN model to use. Options: 'LSTM', 'SimpleRNN', 'Bi-LSTM', or 'Bi-GRU'.
        input_shape (tuple): Shape of input data (sequence_length, num_features).
        num_layers (int): Number of recurrent layers to stack.
        num_units (int): Number of units (neurons) in each recurrent layer.
        recurrent_activation_function (str): Activation function for recurrent layers.
        recurrent_regularization (float): Strength of L2 regularization for recurrent weights.
        kernel_regularization (float): Strength of L2 regularization for kernel weights.
        bias_regularization (float): Strength of L2 regularization for bias terms.
        dropout_rate (float): Dropout rate for dropout layers.
        recurrent_dropout_rate (float): Recurrent dropout rate for recurrent layers.

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
                model.add(LSTM(units=num_units, activation=recurrent_activation_function, use_bias=True,
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=True, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(LSTM(units=num_units, activation=recurrent_activation_function, use_bias=True, 
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=False, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape))

        elif model_type == 'GRU':
            # Add GRU layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(GRU(units=num_units, activation=recurrent_activation_function, use_bias=True,
                              recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                              bias_regularizer=l2(bias_regularization), return_sequences=True, 
                              dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                              input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(GRU(units=num_units, activation=recurrent_activation_function, use_bias=True, 
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=False, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape))

        elif model_type == 'Bi-LSTM':
            # Add Bidirectional LSTM layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(Bidirectional(LSTM(units=num_units, activation=recurrent_activation_function, use_bias=True,
                                             recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                                             bias_regularizer=l2(bias_regularization), return_sequences=True, 
                                             dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate),
                                             input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(Bidirectional(LSTM(units=num_units, activation=recurrent_activation_function, use_bias=True, 
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=False, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape)))

        elif model_type == 'Bi-GRU':
            # Add Bidirectional GRU layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(Bidirectional(GRU(units=num_units, activation=recurrent_activation_function, use_bias=True,
                              recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                              bias_regularizer=l2(bias_regularization), return_sequences=True, 
                              dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate),
                              input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(Bidirectional(GRU(units=num_units, activation=recurrent_activation_function, use_bias=True, 
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=False, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape)))

        elif model_type == 'SimpleRNN':
            # Add SimpleRNN layer
            if i < num_layers - 1:
                # If not the last layer, return sequences
                model.add(SimpleRNN(units=num_units, activation=recurrent_activation_function, use_bias=True,
                                    recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                                    bias_regularizer=l2(bias_regularization), return_sequences=True, 
                                    dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                                    input_shape=input_shape))
            else:
                # If the last layer, do not return sequences
                model.add(SimpleRNN(units=num_units, activation=recurrent_activation_function, use_bias=True, 
                               recurrent_regularizer=l2(recurrent_regularization), kernel_regularizer=l2(kernel_regularization), 
                               bias_regularizer=l2(bias_regularization), return_sequences=False, 
                               dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                               input_shape=input_shape))

        else:
            # Raise an error for invalid model type
            raise ValueError("Invalid model type. Please choose from 'LSTM', 'SimpleRNN', 'Bi-LSTM', or 'Bi-GRU'.")
        
        # Add dropout layer
        if dropout_rate > 0.0:
            model.add(Dropout(rate=dropout_rate))
        
    # Add output Dense layer
    model.add(Dense(units=1))
    
    return model

