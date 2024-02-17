#!/usr/bin/env python
# coding: utf-8

# In[158]:


#get_ipython().system('jupyter nbconvert --to script Transformer.ipynb')


# In[159]:


import numpy as np
import tensorflow as tf
from tensorflow import keras


# **Source of implementation for Positional Encoding:**
# - https://stackoverflow.com/questions/68477306/positional-encoding-for-time-series-based-data-for-transformer-dnn-models
# - https://www.tensorflow.org/text/tutorials/transformer
# - https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
# - https://github.com/antonio-f/PositionalEncoding/blob/main/positional_encoding.py
# 
# ![](../../../images/Positional_Encoding.jpg)

# In[160]:


def positional_encoding(pos, d_model):
    # Calculate angle rates based on dimension indices
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    # Generate a matrix of shape (pos, d_model) containing angle values for each position and dimension
    angle_rads = np.arange(pos)[:, np.newaxis] * angle_rates
    # Apply sine to even indices in the matrix; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cosine to odd indices in the matrix; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # Add a new axis to the matrix to create a 3D tensor and cast to TensorFlow float32 dtype
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


# **Self and Multi-Head Attention**<br><br>
# ![](../../../images/Attention.png)

# **Source of implementation for Scaled Dot Product Attention:** 
# - https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/
# - https://rmoklesur.medium.com/understanding-scaled-dot-product-attention-with-tensorflow-f570245bc12c
# - https://jamesmccaffrey.wordpress.com/2020/09/10/trying-to-understand-scaled-dot-product-attention-for-transformer-architecture/
# - https://gist.github.com/edumunozsala/72d25ca4ef1d5fde7eb4ebbd5d51792f
# 
# ![](../../../images/Self_Attention.jpg)

# In[161]:


# Implementing the Scaled-Dot Product Attention - Self Attention
def scaled_dot_product_attention(queries, keys, values, mask):
    # Calculate the dot product of query and key matrices
    matmul_qk = tf.matmul(queries, keys, transpose_b=True) # Q * K.T
    
    # Get the dimension of the key matrix and cast to float32
    d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
    
    # Scale the attention scores by the square root of the key dimension / Scoring the queries against the keys after transposing the latter, and scaling
    scaled_attention_scores = matmul_qk / tf.math.sqrt(d_k)
    
    # Apply the mask to the attention scores (if mask is provided)
    if mask is not None:
        scaled_attention_scores += (mask * -1e9)
    
    # Computing the weights by a softmax operation
    attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)
    
    # Calculate the output by multiplying attention weights with value matrix
    output = tf.matmul(attention_weights, values)

    return output, attention_weights


# ### **Creating a Transformer Manually Without Embeddings**

# **Source of implementation for Multi-Head Attention:** 
# - https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/
# 
# ![](../../../images/Multi_Head_Att.jpg)

# In[162]:


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, mask):
        # Calculate the dot product of query and key matrices
        matmul_qk = tf.matmul(queries, keys, transpose_b=True) # Q * K.T

        # Get the dimension of the key matrix and cast to float32
        d_k = keys.shape[-1]
        d_k = tf.cast(d_k, tf.float32)

        # Scale the attention scores by the square root of the key dimension / Scoring the queries against the keys after transposing the latter, and scaling
        scaled_attention_scores = matmul_qk / tf.math.sqrt(d_k)

        # Apply the mask to the attention scores (if mask is provided)
        if mask is not None:
            scaled_attention_scores += (mask * -1e9)

        # Computing the weights by a softmax operation
        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)

        # Calculate the output by multiplying attention weights with value matrix
        output = tf.matmul(attention_weights, values)

        return output
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        # Initialize linear layers for projections
        self.query_projection = tf.keras.layers.Dense(units=(h * d_k), activation=None)
        self.key_projection = tf.keras.layers.Dense(units=(h * d_k), activation=None)
        self.value_projection = tf.keras.layers.Dense(units=(h * d_v), activation=None)

        # Initialize the final linear layer
        self.output_projection = tf.keras.layers.Dense(units=d_model, activation=None)

        # Initialize attention layer
        self.attention = ScaledDotProductAttention()

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]

        # Linear projections
        queries = tf.reshape(self.query_projection(queries), (batch_size, -1, self.h, self.d_k))
        keys = tf.reshape(self.key_projection(keys), (batch_size, -1, self.h, self.d_k))
        values = tf.reshape(self.value_projection(values), (batch_size, -1, self.h, self.d_v))

        # Transpose to have dimensions [batch_size, num_heads, seq_len, d_k/d_v]
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        # Apply attention
        attention_output = self.attention(queries, keys, values, mask)

        # Transpose and concatenate to get the final output
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.h * self.d_v))

        # Apply final linear layer
        output = self.output_projection(attention_output)

        return output


# **Source of implementation for Position-wise-Feed-Forward:**
# - https://stackoverflow.com/questions/74979359/how-is-position-wise-feed-forward-neural-network-implemented-for-transformers
# 
# ![](../../../images/Positionwise_FeedForward.jpg)

# In[163]:


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)

        # Feedforward neural network with a ReLU activation
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(d_ff, activation='relu'),tf.keras.layers.Dense(d_model, activation=None)])

        # Dropout layer to prevent overfitting
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        # Pass the inputs through the feedforward neural network
        ff_output = self.ffn(inputs)

        # Apply dropout to the output
        ff_output = self.dropout(ff_output)

        return ff_output


# **Source of implementation for _Encoder_:**
# - https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/
# - https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
# - https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_10_5_keras_transformers.ipynb
# 
# ![](../../../images/Encoder.jpg)

# In[164]:


def Encoder(encoder_input, num_heads, d_ff, dropout_rate, encoder_mask):
    inputs = encoder_input
    
    # Extract the size of the model from the input shape
    d_model = inputs.shape[-1]

    # Multi-Head Self Attention
    attention_output = MultiHeadAttention(h=num_heads,
                                 d_k=d_model // num_heads,
                                 d_v=d_model // num_heads,
                                 d_model=d_model)(inputs, inputs, inputs, mask=encoder_mask)
    # Apply dropout for regularization
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)

    # Add and Normalize step after Multi-Head Self Attention
    norm_attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feedforward Neural Network
    ffn = PositionwiseFeedForward(
        d_model=d_model,
        d_ff=d_ff,
        dropout_rate=dropout_rate
    )
    ff_output = ffn(norm_attention_output)
    # Apply dropout for regularization
    ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)

    # Add and Normalize step after the Feedforward Neural Network
    encoder_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(norm_attention_output + ff_output)

    return encoder_output


# **Source of implementation for _Decoder_:**
# - https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/
# 
# ![](../../../images/Decoder.jpg)

# In[165]:


def Decoder(inputs, encoder_output, num_heads, d_ff, dropout_rate, decoder_mask):
    inputs = inputs

    # Extract the size of the model from the input shape
    d_model = inputs.shape[-1]

    # Masked Self-Attention
    masked_attention_output = MultiHeadAttention(
                                                 h=num_heads,
                                                 d_k=d_model // num_heads,
                                                 d_v=d_model // num_heads,
                                                 d_model=d_model
                                                )(inputs, inputs, inputs, mask=decoder_mask)
    # Apply dropout for regularization
    masked_attention_output = tf.keras.layers.Dropout(dropout_rate)(masked_attention_output)

    # Add and Normalize the Masked Self-Attention output
    norm_masked_attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(masked_attention_output + inputs)

    # Cross-Attention with Encoder Output
    attention_output = MultiHeadAttention(
                                          h=num_heads,
                                          d_k=d_model // num_heads,
                                          d_v=d_model // num_heads,
                                          d_model=d_model
                                         )(norm_masked_attention_output, encoder_output, encoder_output, mask=decoder_mask)  # < ----- try and mask = mask
    # Apply dropout for regularization
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)

    # Add and Normalize the Cross-Attention output
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(norm_masked_attention_output + attention_output)

    # Feedforward Neural Network
    ffn = PositionwiseFeedForward(
                                  d_model=d_model,
                                  d_ff=d_ff,
                                  dropout_rate=dropout_rate
                                 )
    ff_output = ffn(attention_output)
    # Apply dropout for regularization
    ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)

    # Add and Normalize
    decoder_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    return decoder_output


# **Source of implementation for _Transformer Model_:**
# - https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/
# 
# ![](../../../images/Transformer_model.jpg)

# In[166]:


def TransformerModel(input_shape, num_heads, d_ff, num_layers, dropout_rate, encoder_mask, decoder_mask):
    # Define the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    encoder = inputs
    for _ in range(num_layers):
        # Apply the Encoder function to the input for each layer
        encoder_output = Encoder(encoder, num_heads, d_ff, dropout_rate, encoder_mask)

    # Decoder
    decoder = encoder_output
    for _ in range(num_layers):
        # Apply the Decoder function to the encoder output for each layer
        decoder_output = Decoder(decoder, encoder, num_heads, d_ff, dropout_rate, decoder_mask)

    # Generate the final output with a TimeDistributed Dense layer
    pull_time_window = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(decoder_output)
    pull_time_window = tf.keras.layers.Dropout(0.1)(pull_time_window)
    outputs = tf.keras.layers.Dense(1, activation='linear')(pull_time_window)
    #outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))(decoder_output)

    # Build the Keras model using the specified inputs and outputs
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# In[167]:


def main(sequence_length, num_features, num_heads, d_ff, num_layers, dropout_rate, encoder_mask, decoder_mask):
    # Define the input_shape
    input_shape = (sequence_length, num_features)
    # Create the transformer model
    manual_model = TransformerModel(input_shape, num_heads, d_ff, num_layers, dropout_rate, encoder_mask, decoder_mask)

    # Print the model summary
    manual_model.summary()

    return manual_model


# ****
# ****

# ### **Create a Transformer with keras**

# In[168]:


def keras_transformer_model(input_shape, num_heads, dff, num_layers, dropout_rate):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    encoder = inputs
    # encoder = layers.Dense(units=input_shape[0], activation="relu")(inputs) # < ------
    for i in range(num_layers):
        # Multi-Head Self Attention
        encoder = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[0])(encoder, encoder)
        encoder = tf.keras.layers.Dropout(dropout_rate)(encoder)
        # Add and Normalize
        encoder = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoder)
        # Feedforward
        ffn = keras.Sequential([tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(input_shape[0])])
        encoder = ffn(encoder)
        encoder = tf.keras.layers.Dropout(dropout_rate)(encoder)
        # Add and Normalize
        encoder = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoder)
        
    # Decoder
    decoder = encoder
    # decoder = layers.Dense(units=input_shape[0], activation="relu")(inputs)
    for i in range(num_layers):
        # Masked Multi-Head Attention
        decoder = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[0])(decoder, decoder)
        decoder = tf.keras.layers.Dropout(dropout_rate)(decoder)
        # Add and Normalize the masked multi-head attention
        decoder = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder)
        # Multi-Head Attention
        decoder = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[0])(decoder, encoder) # < ------
        decoder = tf.keras.layers.Dropout(dropout_rate)(decoder) # < ------
        # Add and Normalize the multi-head attention
        decoder = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder) # < ------
        # Feedforward
        ffn = keras.Sequential([tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(input_shape[0])])
        decoder = ffn(decoder)
        decoder = tf.keras.layers.Dropout(dropout_rate)(decoder)
        # Add and Normalize
        decoder = tf.keras.layers.LayerNormalization(epsilon=1e-6)(decoder)

    outputs = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(decoder)
    outputs = tf.keras.layers.Dropout(0.1)(outputs)
    outputs = tf.keras.layers.Dense(1, activation='linear')(outputs)
    #outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))(decoder)
    #outputs = tf.keras.layers.Dense(units=input_shape[0])(decoder) # < ------
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

