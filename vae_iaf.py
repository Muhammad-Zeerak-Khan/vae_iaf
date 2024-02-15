import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

class SamplingLayer(tf.keras.layers.Layer):
    """Sampling layer with Inverse Autoregressive Flow (IAF) for VAE."""
    
    def __init__(self, latent_dim, num_flows=1, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        self.base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))
        
        # IAF Bijectors
        self.iaf_bijectors = [
            tfp.bijectors.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfp.bijectors.AutoregressiveNetwork(params=2, hidden_units=[256, 256])
            ) for _ in range(num_flows)
        ]
        self.chain_of_flows = tfp.bijectors.Chain(list(reversed(self.iaf_bijectors))) # Reverse the order for correct application

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Reparameterization trick
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Apply IAF
        flow_dist = tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain_of_flows)
        z = flow_dist.bijector.forward(z)
        return z
def create_encoder(latent_dim, num_flows):
    inputs = tfkl.Input(shape=(28, 28, 1))
    x = tfkl.Flatten()(inputs)
    x = tfkl.Dense(512, activation='relu')(x)
    z_mean = tfkl.Dense(latent_dim)(x)
    z_log_var = tfkl.Dense(latent_dim)(x)
    
    z = SamplingLayer(latent_dim, num_flows)([z_mean, z_log_var])
    
    encoder = tfk.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder

def create_decoder(latent_dim):
    inputs = tfkl.Input(shape=(latent_dim,))
    x = tfkl.Dense(7*7*32, activation='relu')(inputs)
    x = tfkl.Reshape((7, 7, 32))(x)
    x = tfkl.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = tfkl.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = tfkl.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    
    decoder = tfk.Model(inputs, outputs, name='decoder')
    return decoder
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, num_flows):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = create_encoder(latent_dim, num_flows)
        self.decoder = create_decoder(latent_dim)
        
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstructed = self.decoder(z)
            
            # Reconstruction Loss
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, reconstructed)) * 28 * 28
            
            # KL Divergence Loss
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            # Total Loss
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


latent_dim = 2
num_flows = 2

vae = VAE(latent_dim=latent_dim, num_flows=num_flows)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# Assuming X_train is your dataset
# history = vae.fit(X_train, epochs=30, batch_size=128)


