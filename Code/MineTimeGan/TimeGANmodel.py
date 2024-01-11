#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().system('jupyter nbconvert --to script sequences.ipynb')
# get_ipython().system('jupyter nbconvert --to script ModelBuildingFunctions.ipynb')
# get_ipython().system('jupyter nbconvert --to script LossFunctions.ipynb')


# In[1]:


import numpy as np
import tensorflow as tf
from sequences import time_series_to_sequences, sequences_to_time_series
from ModelBuildingFunctions import encoder_embedder, encoder, decoder, generator_embedder, generator, discriminator, simulator
from LossFunctions import binary_crossentropy, mean_squared_error


# In[2]:


class TimeGAN():
    def __init__(self,
                 x,
                 timesteps,
                 hidden_dim,
                 num_layers,
                 lambda_param,
                 eta_param,
                 learning_rate,
                 batch_size):
        '''
        Implementation of synthetic time series generation model introduced in Yoon, J., Jarrett, D. and Van der Schaar, M., 2019.
        Time-series generative adversarial networks. Advances in neural information processing systems, 32.
        '''
        
        # Ensure x is a two-dimensional array
        if len(x.shape) == 1:
            x = np.reshape(x, (-1, 1))

        # extract the length of the time series
        samples = x.shape[0]

        # extract the number of time series
        features = x.shape[1]

        # scale the time series
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        x = (x - mu) / sigma

        # reshape the time series as sequences
        x = time_series_to_sequences(time_series=x, timesteps=timesteps)
        
        # create the dataset
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.cache().shuffle(samples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        # build the models
        autoencoder_model = tf.keras.models.Sequential([
            encoder_embedder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=1),
            encoder(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers - 1),
            decoder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=num_layers)
        ])
    
        generator_model = tf.keras.models.Sequential([
            generator_embedder(timesteps=timesteps, features=features, hidden_dim=hidden_dim, num_layers=1),
            generator(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers - 1),
        ])
        
        discriminator_model = discriminator(timesteps=timesteps, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # instantiate the optimizers
        autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # save the objects
        self.mu = mu
        self.sigma = sigma
        self.samples = samples
        self.timesteps = timesteps
        self.features = features
        self.lambda_param = lambda_param
        self.eta_param = eta_param
        self.dataset = dataset
        self.autoencoder_model = autoencoder_model
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.autoencoder_optimizer = autoencoder_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
    
    def fit(self, epochs, verbose=True):
        '''
        Train the model.
        '''
        
        def train_step(data):
            with tf.GradientTape() as autoencoder_tape, tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                
                # get the actual sequences
                x = tf.cast(data, dtype=tf.float32)
                
                # generate the synthetic sequences
                z = simulator(samples=x.shape[0], timesteps=self.timesteps, features=self.features)

                # get the encoder outputs
                ex = self.autoencoder_model.get_layer('encoder_embedder')(x)     # actual embedding
                hx = self.autoencoder_model.get_layer('encoder')(ex)             # actual latent vector

                # get the generator outputs
                ez = self.generator_model.get_layer('generator_embedder')(z)     # synthetic embedding
                hz = self.generator_model.get_layer('generator')(ez)             # synthetic latent vector
                hx_hat = self.generator_model.get_layer('generator')(ex)         # conditional synthetic latent vector (i.e. given the actual embedding)
                
                # get the decoder outputs
                x_hat = self.autoencoder_model.get_layer('decoder')(hx)          # reconstructed sequences

                # get the discriminator outputs
                p_ex = self.discriminator_model(ex)                              # log-odds of actual embedding
                p_ez = self.discriminator_model(ez)                              # log-odds of synthetic embedding
                p_hx = self.discriminator_model(hx)                              # log-odds of actual latent vector
                p_hz = self.discriminator_model(hz)                              # log-odds of synthetic latent vector

                # calculate the supervised loss
                supervised_loss = mean_squared_error(hx[:, 1:, :], hx_hat[:, :-1, :])
                
                # calculate the autoencoder loss
                autoencoder_loss = mean_squared_error(x, x_hat) +                                    self.lambda_param * supervised_loss
                                   
                # calculate the generator loss
                generator_loss = binary_crossentropy(tf.ones_like(p_hz), p_hz) +                                  binary_crossentropy(tf.ones_like(p_ez), p_ez) +                                  self.eta_param * supervised_loss

                # calculate the discriminator loss
                discriminator_loss = binary_crossentropy(tf.zeros_like(p_hz), p_hz) +                                      binary_crossentropy(tf.zeros_like(p_ez), p_ez) +                                      binary_crossentropy(tf.ones_like(p_hx), p_hx) +                                      binary_crossentropy(tf.ones_like(p_ex), p_ex)
            
            # calculate the gradients
            autoencoder_gradient = autoencoder_tape.gradient(autoencoder_loss, self.autoencoder_model.trainable_variables)
            generator_gradient = generator_tape.gradient(generator_loss, self.generator_model.trainable_variables)
            discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator_model.trainable_variables)
            
            # update the weights
            self.autoencoder_optimizer.apply_gradients(zip(autoencoder_gradient, self.autoencoder_model.trainable_variables))
            self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator_model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator_model.trainable_variables))
            
            return autoencoder_loss, generator_loss, discriminator_loss

        # train the model
        for epoch in range(epochs):
            for data in self.dataset:
                autoencoder_loss, generator_loss, discriminator_loss = train_step(data)
            if verbose:
                print(
                    f'epoch: {1 + epoch} '
                    f'autoencoder_loss: {format(autoencoder_loss.numpy(), ".6f")} '
                    f'generator_loss: {format(generator_loss.numpy(), ".6f")} '
                    f'discriminator_loss: {format(discriminator_loss.numpy(), ".6f")}'
                )

    def reconstruct(self, x):
        '''
        Reconstruct the time series.
        '''
        
        # scale the time series
        x = (x - self.mu) / self.sigma

        # reshape the time series as sequences
        x = time_series_to_sequences(time_series=x, timesteps=self.timesteps)

        # get the reconstructed sequences
        x_hat = self.autoencoder_model(x)
        
        # transform the reconstructed sequences back to time series
        x_hat = sequences_to_time_series(x_hat.numpy())
   
        # transform the reconstructed time series back to the original scale
        x_hat = self.mu + self.sigma * x_hat
        
        return x_hat
    
    def simulate(self, samples, forecast_steps):
        '''
        Simulate the time series, including forecasting.
        '''
        
        # generate the synthetic sequences
        z = simulator(samples=(samples + forecast_steps) // self.timesteps, timesteps=self.timesteps, features=self.features)
        
        # get the simulated sequences
        x_sim = self.autoencoder_model.get_layer('decoder')(self.generator_model(z))
        
        # extract the forecasting part
        x_forecast = x_sim[:, -forecast_steps:, :]

        # transform the simulated sequences back to time series
        x_sim = sequences_to_time_series(x_sim.numpy())
        
        # transform the simulated time series back to the original scale
        x_sim = self.mu + self.sigma * x_sim
        
        return x_sim, x_forecast


# ## **Core Concept: Generative Adversarial Network (GAN)**
# 
# TimeGAN employs a Generative Adversarial Network (GAN) framework, consisting of a generator ($G$) and a discriminator ($D$). The generator learns to produce synthetic time series sequences ($X_{\text{synthetic}}$) that are indistinguishable from real sequences ($X_{\text{real}}$), while the discriminator aims to differentiate between real and synthetic data.
# 
# ## **Model Components:**
# 
# 1. ### **Encoder-Embedder** (**E**<sub>embed</sub> and **E**):
# 
#     **Encoder-Embedder Function:**  
#     - **E<sub>embed</sub>:** **X**<sub>real</sub> → **E**<sub>embed</sub>(**X**<sub>real</sub>)  
#    
#     **Encoder Function:**  
#     - **E:** **E**<sub>embed</sub>(**X**<sub>real</sub>) → **H**<sub>real</sub>
# 
# 2. ### **Decoder** (**D**):
# 
#     **Decoder Function:**  
#     - **D:** **H**<sub>real</sub> → **X**<sub>reconstructed</sub>
# 
# 3. ### **Generator-Embedder** (**G**<sub>embed</sub> and **G**):
# 
#     **Generator-Embedder Function:**  
#     - **G<sub>embed</sub>:** **Z**<sub>synthetic</sub> → **G**<sub>embed</sub>(**Z**<sub>synthetic</sub>)  
#      
#      **Generator Function:**  
#     - **G:** **G**<sub>embed</sub>(**Z**<sub>synthetic</sub>) → **H**<sub>synthetic</sub>
# 
# 4. ### **Discriminator** (**D**):
# 
#     **Discriminator Function:**  
#     - **D:** **H**<sub>real</sub> ∪ **H**<sub>synthetic</sub> → **P**<sub>discriminate</sub>
# 
# ## **Loss Functions:**
# 
# 1. ### **Supervised Loss** (**L**<sub>supervised</sub>): 
#     - **L<sub>supervised</sub>:** MeanSquaredError(**H**<sub>real</sub>, **H**<sub>synthetic</sub>)
# 
# 2. ### **Autoencoder Loss** (**L**<sub>autoencoder</sub>):
#     - **L<sub>autoencoder</sub>:** MeanSquaredError(**X**<sub>real</sub>, **X**<sub>reconstructed</sub>) + **λ** ⋅ **L**<sub>supervised</sub>
# 
# 
# 3. ### **Generator Loss** (**L**<sub>generator</sub>):
#     - **L<sub>generator</sub>:** BinaryCrossEntropy(Ones, **P**<sub>discriminate</sub>(**H**<sub>synthetic</sub>)) + **η** ⋅ **L**<sub>supervised</sub>
# 
# 4. ### **Discriminator Loss** (**L**<sub>discriminator</sub>):
#     - **L<sub>discriminator</sub>:** BinaryCrossEntropy(Zeros, **P**<sub>discriminate</sub>(**H**<sub>real</sub>)) +<br> BinaryCrossEntropy(Zeros, **P**<sub>discriminate</sub>(**H**<sub>synthetic</sub>)) +<br> BinaryCrossEntropy(Ones, **P**<sub>discriminate</sub>(**H**<sub>real</sub>)) +<br> BinaryCrossEntropy(Ones, **P**<sub>discriminate</sub>(**H**<sub>synthetic</sub>))
# 
# ## **Simulator Function: Wiener Process**
# 
# - **Z**<sub>synthetic</sub> ∼ **N**(0,1)  
# - **Z**<sub>synthetic</sub>: Cumsum(**Z**<sub>synthetic</sub>) / **T**, where **T** is the total number of time steps.
# 
# ## **Normalization:**
# - **MinMaxScaler**
# - **X**<sub>synthetic</sub>: (**X**<sub>synthetic</sub> − Mean(**X**<sub>synthetic</sub>)) / Std(**X**<sub>synthetic</sub>)
# 
# ## **Training Loop:**
# 
# - Iteratively optimize model parameters using gradient descent based on the calculated losses.
# 
# ## **Conclusion:**
# 
# TimeGAN combines supervised and unsupervised objectives, leveraging GAN principles to generate realistic time series data. The model undergoes training to minimize reconstruction errors, adversarial losses, and ensure that the synthetic data closely resembles real-world sequences. For a comprehensive derivation and detailed mathematical insights, please refer to the original paper, "Time-series Generative Adversarial Networks" by Yoon, Jarrett, and Van der Schaar.

# %%
