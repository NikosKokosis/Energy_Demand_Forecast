#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('jupyter nbconvert --to script utils.ipynb')


# In[2]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


def split_dataset(df, train_ratio, val_ratio):

    total_size = len(df)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    assert len(train_df) + len(val_df) + len(test_df) == total_size, "Dataset not split correctly."

    print(f'Training split ratio:   {round(len(train_df) / len(df), 3)}')
    print(f'Validation split ratio: {round(len(val_df) / len(df), 3)}')
    print(f'Testing split ratio:    {round(len(test_df) / len(df), 3)}')
    print("\nShapes of the datasets:")
    print(train_df.shape, val_df.shape, test_df.shape)

    return train_df, val_df, test_df


# In[4]:


def create_sequences(data, sequence_length):
    inputs = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i + sequence_length].values
        target = data.iloc[i + sequence_length]['Energy__kWh_']  # Predict the next value
        inputs.append(sequence)
        targets.append(target)

    inputs_array = np.array(inputs)
    targets_array = np.array(targets)
    
    print(f'Dataset split into sequences:')
    print(f'Sequences shape: {inputs_array.shape}')
    print(f'Targets shape: {targets_array.shape}\n')

    return np.array(inputs), np.array(targets)


# In[5]:


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(
        tf.keras.backend.mean(
            tf.keras.backend.square(
                y_pred - y_true
            )
        ) + 1e-9
    )


# In[6]:


def plot_metrics(history):
    fig = plt.figure(figsize=(15,20))

    # Plot model loss
    ax1 = fig.add_subplot(311)
    ax1.plot(history.history['loss'], label='Training loss (MSE)')
    ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12)

    # Plot MAE
    ax2 = fig.add_subplot(312)
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title("Model metric - Mean Absolute Error (MAE)", fontsize=18)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.legend(loc="best", fontsize=12)

    # Plot RMSE
    ax3 = fig.add_subplot(313)
    ax3.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    ax3.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    ax3.set_title("Model metric - Root Mean Squared Error (RMSE)", fontsize=18)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Root Mean Squared Error (RMSE)')
    ax3.legend(loc="best", fontsize=12)

    plt.tight_layout()
    plt.show()

