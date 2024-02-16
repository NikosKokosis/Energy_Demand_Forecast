#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('jupyter nbconvert --to script RNN_utils.ipynb')


# In[21]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error


# In[22]:


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


# In[23]:


def one_hot_encode(df, months_col, weekdays_col):

    # Set all month and weekday columns to 0
    df[months_col + weekdays_col] = 0

    # Set values for months
    for month in months_col:
        df.loc[df['Month'] == month, month] = 1

    # Set values for weekdays
    for weekday in weekdays_col:
        df.loc[df['Weekday'] == weekday, weekday] = 1

    return df


# In[24]:


def one_hot_months(df):
    # List of all months
    months_col = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    # Set all month and weekday columns to 0
    df[months_col] = 0

    # Set values for months
    for month in months_col:
        df.loc[df['Month'] == month, month] = 1

    return df


# In[25]:


def create_sequences(df, sequence_length, target_column, step):
    sequences = []
    targets = []

    # Generate sequences and targets/labels
    for i in range(0, len(df) - sequence_length, step):
        sequence = df.iloc[i: i + sequence_length]
        target = df.iloc[i + sequence_length][target_column]
        sequences.append(sequence)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)
    print(f'Dataset split into sequences:')
    print(f'Sequences shape: {sequences.shape}')
    print(f'Targets shape: {targets.shape}\n')

    return sequences, targets


# In[26]:


def scale_data_with_fitted_scaler(scaler, X, Y, numerical_D):
    # ----------------------
    #    SCALE THE X_array with the fitted scaler of X_train
    # ----------------------
    # Extract the first 5 columns from X
    X_subset = X[:, :, :numerical_D]

    # Reshape the subset to 2D array
    X_reshaped = X_subset.reshape(-1, numerical_D)

    # Scaling 
    X_scaled_subset = scaler.transform(X_reshaped)

    X_scaled_subset = X_scaled_subset.reshape(X.shape[0], X.shape[1], numerical_D)

    # Combine the scaled subset with the remaining columns of X
    X_scaled = np.concatenate([X_scaled_subset, X[:, :, 5:]], axis=-1)

    # ----------------------
    #    SCALE THE Y_array with the fitted scaler of X_train
    # ----------------------

    # Reshape the subset of Y to 2D array
    Y_reshaped = Y.reshape(-1,1)

    # Add extra columns of zeros to match the expected dimension for concatenation
    Y_reshaped = np.concatenate((Y_reshaped, np.zeros((Y_reshaped.shape[0], numerical_D-1))), axis=1)

    # Use the previously fitted scaler to transform the Y_reshaped
    Y_scaled = scaler.transform(Y_reshaped)

    # Keep only the energy column (target) in the Y_scaled
    Y_scaled = Y_scaled[:,:1]

    return X_scaled, Y_scaled


# In[27]:


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(
        tf.keras.backend.mean(
            tf.keras.backend.square(
                y_pred - y_true
            )
        ) + 1e-9
    )


# In[28]:


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[29]:


def evaluate_predictions_model(model, model_name, X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test, scaler):
    # Predictions on training data
    y_pred_train = model.predict(X_train_scaled)
    inversed_y_pred_train = scaler.inverse_transform(np.concatenate([y_pred_train, np.zeros((y_pred_train.shape[0], scaler.n_features_in_-1))], axis=1))
    inversed_y_pred_train = inversed_y_pred_train[:, 0]

    # Metrics on training data
    train_rmse = calculate_rmse(Y_train, inversed_y_pred_train)
    train_mae = calculate_mae(Y_train, inversed_y_pred_train)

    # Predictions on validation data
    y_pred_val = model.predict(X_val_scaled)
    inversed_y_pred_val = scaler.inverse_transform(np.concatenate([y_pred_val, np.zeros((y_pred_val.shape[0], scaler.n_features_in_-1))], axis=1))
    inversed_y_pred_val = inversed_y_pred_val[:, 0]

    # Metrics on validation data
    val_rmse = calculate_rmse(Y_val, inversed_y_pred_val)
    val_mae = calculate_mae(Y_val, inversed_y_pred_val)

    # Predictions on test data
    y_pred_test = model.predict(X_test_scaled)
    inversed_y_pred_test = scaler.inverse_transform(np.concatenate([y_pred_test, np.zeros((y_pred_test.shape[0], scaler.n_features_in_-1))], axis=1))
    inversed_y_pred_test = inversed_y_pred_test[:, 0]

    # Metrics on test data
    test_rmse = calculate_rmse(Y_test, inversed_y_pred_test)
    test_mae = calculate_mae(Y_test, inversed_y_pred_test)
    
    # Print the results
    print(f"\n\nEvaluation metrics for {model_name} model:\n-------------------")
    print('Train Dataset:')
    print(f"RMSE: {train_rmse}")
    print(f"MAE: {train_mae}\n-------------------")

    print('Validation Dataset:')
    print(f"RMSE: {val_rmse}")
    print(f"Validation MAE: {val_mae}\n-------------------")
    
    print('Test Dataset:')
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}\n\n")

