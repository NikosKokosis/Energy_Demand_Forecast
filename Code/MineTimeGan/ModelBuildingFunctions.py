#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# 1. `encoder_embedder`: This function defines the encoder embedder, which takes the actual sequences as input and returns the actual embeddings. The encoder embedder is responsible for mapping the input sequences to a lower-dimensional latent space. The authors use a GRU layer to process the input sequences and generate the embeddings. The output of the encoder embedder is a sequence of embeddings that capture the temporal dynamics of the input sequences.

# In[ ]:


def encoder_embedder(timesteps, features, hidden_dim, num_layers):
    '''
    Encoder embedder, takes as input the actual sequences and returns the actual embeddings.
    '''
    x = tf.keras.layers.Input(shape=(timesteps, features))
    for _ in range(num_layers):
        e = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(x if _ == 0 else e)
    return tf.keras.models.Model(x, e, name='encoder_embedder')


# 2. `encoder`: This function defines the encoder, which takes the actual embeddings as input and returns the actual latent vector. The encoder is responsible for further reducing the dimensionality of the embeddings and generating a compact representation of the input sequences. The authors use GRU layers to process the input embeddings and generate the latent vector. The output of the encoder is a single vector that summarizes the temporal dynamics of the input sequences.

# In[2]:


def encoder(timesteps, hidden_dim, num_layers):
    '''
    Encoder, takes as input the actual embeddings and returns the actual latent vector.
    '''
    e = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        h = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(e if _ == 0 else h)
    h = tf.keras.layers.Dense(units=hidden_dim)(h)
    return tf.keras.models.Model(e, h, name='encoder')


# 3. `decoder`: This function defines the decoder, which takes the actual or synthetic latent vector as input and returns the reconstructed or synthetic sequences. The decoder is responsible for mapping the latent vector back to the original feature space. The authors use TimeDistributed dense layers to reconstruct the sequences from the latent vector. The output of the decoder is a sequence of reconstructed or synthetic sequences that match the temporal dynamics of the input sequences.

# In[ ]:


def decoder(timesteps, features, hidden_dim, num_layers):
    '''
    Decoder, takes as input the actual or synthetic latent vector and returns the reconstructed or synthetic sequences.
    '''
    h = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=hidden_dim, activation='relu'))(h if _ == 0 else y)
    y = tf.keras.layers.Dense(units=features)(y)
    return tf.keras.models.Model(h, y, name='decoder')


# 4. `generator_embedder`: This function defines the generator embedder, which takes the synthetic sequences as input and returns the synthetic embeddings. The generator embedder is responsible for mapping the synthetic sequences to the same lower-dimensional latent space as the actual sequences. The authors use GRU layers to process the input synthetic sequences and generate the synthetic embeddings. The output of the generator embedder is a sequence of embeddings that capture the temporal dynamics of the synthetic sequences.

# In[3]:


def generator_embedder(timesteps, features, hidden_dim, num_layers):
    '''
    Generator embedder, takes as input the synthetic sequences and returns the synthetic embeddings.
    '''
    z = tf.keras.layers.Input(shape=(timesteps, features))
    for _ in range(num_layers):
        e = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(z if _ == 0 else e)
    return tf.keras.models.Model(z, e, name='generator_embedder')


# 5. `generator`: This function defines the generator, which takes the synthetic embeddings as input and returns the synthetic latent vector. The generator is responsible for generating synthetic latent vectors that match the temporal dynamics of the actual latent vectors. The authors use GRU layers to process the input embeddings and generate the synthetic latent vector. The output of the generator is a single vector that summarizes the temporal dynamics of the synthetic sequences.

# In[ ]:


def generator(timesteps, hidden_dim, num_layers):
    '''
    Generator, takes as input the synthetic embeddings and returns the synthetic latent vector.
    '''
    e = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        h = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True)(e if _ == 0 else h)
    h = tf.keras.layers.Dense(units=hidden_dim)(h)
    return tf.keras.models.Model(e, h, name='generator')


# 6. `discriminator`: This function defines the discriminator, which takes the actual or synthetic embedding or latent vector as input and returns the log-odds. The discriminator is responsible for distinguishing between actual and synthetic sequences. The authors use Bidirectional GRU layers to process the input embeddings or latent vectors and make the discrimination. The output of the discriminator is a scalar value that represents the probability that the input sequence is real.

# In[4]:


def discriminator(timesteps, hidden_dim, num_layers):
    '''
    Discriminator, takes as input the actual or synthetic embedding or latent vector and returns the log-odds.
    '''
    h = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    for _ in range(num_layers):
        p = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=hidden_dim, return_sequences=True if _ < num_layers - 1 else False))(h if _ == 0 else p)
    p = tf.keras.layers.Dense(units=1)(p)
    return tf.keras.models.Model(h, p, name='discriminator')


# 7. `simulator`: This function generates synthetic sequences from a Wiener process. The simulator is used to generate synthetic data for training the TimeGAN model. The authors use a random normal distribution to generate samples and then apply a cumulative sum and normalization to obtain synthetic sequences. The output of the simulator is a sequence of synthetic data that matches the temporal dynamics of the actual data.

# In[ ]:


def simulator(samples, timesteps, features):
    '''
    Simulator, generates synthetic sequences from a Wiener process.
    '''
    z = tf.random.normal(mean=0, stddev=1, shape=(samples * timesteps, features), dtype=tf.float32)
    z = tf.cumsum(z, axis=0) / tf.sqrt(tf.cast(samples * timesteps, dtype=tf.float32))
    z = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    z = tf.reshape(z, (samples, timesteps, features))
    return z


# Overall, these functions define the architecture of the TimeGAN model and are used to train the model to generate realistic time-series data. The encoder embedder and encoder are used to map the input sequences to a lower-dimensional latent space, while the generator embedder and generator are used to generate synthetic sequences in the same latent space. The decoder is used to map the latent vectors back to the original feature space, and the discriminator is used to distinguish between actual and synthetic sequences. The simulator is used to generate synthetic data for training the model.
