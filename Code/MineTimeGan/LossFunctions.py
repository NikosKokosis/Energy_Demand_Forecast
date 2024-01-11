#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# **Define two loss functions commonly used in training GANs and autoencoders: mean squared error (MSE) and binary cross-entropy. These functions are used to calculate the supervised loss, reconstruction loss, and unsupervised loss in the context of the TimeGAN model.**

# 1. `mean_squared_error(y_true, y_pred)`: This function calculates the mean squared error between the true values `y_true` and the predicted values `y_pred`. It is commonly used for calculating the supervised loss and the reconstruction loss in the TimeGAN model. The function uses TensorFlow's `tf.keras.losses.mean_squared_error` to compute the element-wise squared error between `y_true` and `y_pred`, and then takes the mean of the sum of these squared errors across the last axis.

# In[ ]:


def mean_squared_error(y_true, y_pred):
    '''
    Mean squared error, used for calculating the supervised loss and the reconstruction loss.
    '''
    loss = tf.keras.losses.mean_squared_error(y_true=tf.expand_dims(y_true, axis=-1), y_pred=tf.expand_dims(y_pred, axis=-1))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# 2. `binary_crossentropy(y_true, y_pred)`: This function calculates the binary cross-entropy between the true values `y_true` and the predicted values `y_pred`. It is commonly used for calculating the unsupervised loss in the TimeGAN model. The function uses TensorFlow's `tf.keras.losses.binary_crossentropy` to compute the binary cross-entropy loss between `y_true` and `y_pred`, and then takes the mean of the loss.

# In[ ]:


def binary_crossentropy(y_true, y_pred):
    '''
    Binary cross-entropy, used for calculating the unsupervised loss.
    '''
    loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
    return tf.reduce_mean(loss)


# **These loss functions are essential for training the TimeGAN model and are used to guide the adversarial learning and reconstruction of the input sequences.**
