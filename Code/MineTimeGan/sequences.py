#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# **These are utility functions that are not directly related to the TimeGAN architecture itself, but rather used to preprocess the time-series data before feeding it into the model.**

# In[ ]:


def time_series_to_sequences(time_series, timesteps):
    '''
    Reshape the time series as sequences.
    '''
    # Ensure time_series is a NumPy array
    time_series = np.array(time_series)

    sequences = []
    for t in range(timesteps, len(time_series) + 1):
        sequence = time_series[t - timesteps: t]
        sequences.append(sequence)

    return np.array(sequences)


def sequences_to_time_series(sequences):
    '''
    Reshape the sequences as time series.
    '''
    time_series = np.concatenate([sequence for sequence in sequences], axis=0)
    return time_series


# The `time_series_to_sequences` function reshapes the time-series data into sequences of a specified length (timesteps). This is done by iterating over the time-series data and creating a sequence of length `timesteps` at each time step. The resulting sequences are then returned as a NumPy array.
# 
# The `sequences_to_time_series` function reshapes the sequences back into a time-series format. This is done by concatenating all the sequences together along the time axis. The resulting time-series data is then returned as a NumPy array.
# 
# These functions are used to preprocess the data before training the TimeGAN model, as the model requires input data in the form of sequences. The authors mention in the paper that they use a sliding window approach to create sequences of length `T` from the original time-series data. This is likely the same approach used in the `time_series_to_sequences` function
