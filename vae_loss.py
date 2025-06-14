import tensorflow as tf
from tensorflow import keras
from keras import layers

class VAEDecoder(tf.keras.Model):
    def __init__(self, latent_dim=512, seq_len=20, output_dim=25):  # 明确输出维度
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        self.fc_mu = keras.Sequential([
            layers.Dense(seq_len * output_dim),
            layers.LayerNormalization(epsilon=1e-6)
        ])
        self.fc_logvar = keras.Sequential([
            layers.Dense(seq_len * output_dim),
            layers.Activation('tanh'),
            layers.LayerNormalization(epsilon=1e-6)
        ])

    def build(self, input_shape):
        super().build(input_shape)
        self.fc_mu.build(input_shape)
        self.fc_logvar.build(input_shape)

    def call(self, theta_1):
        mu = self.fc_mu(theta_1)
        logvar = self.fc_logvar(theta_1) * 0.5  # 压缩输出范围
        eps = tf.random.normal(shape=tf.shape(mu))
        x_recon = mu + tf.exp(logvar) * eps
        x_recon = tf.reshape(x_recon, [-1, self.seq_len, self.output_dim])
        return x_recon, mu, logvar
    
class VAEBridge(tf.keras.Model):
    def __init__(self, input_dim=256, latent_dim=512):
        super().__init__()
        self.fc_mu = keras.Sequential([
            layers.Dense(latent_dim),
            layers.LayerNormalization(epsilon=1e-6)
        ])
        self.fc_logvar = keras.Sequential([
            layers.Dense(latent_dim),
            layers.Activation('tanh'),
            layers.LayerNormalization(epsilon=1e-6)
        ])

    def build(self, input_shape):
        super().build(input_shape)
        self.fc_mu.build(input_shape)
        self.fc_logvar.build(input_shape)

    def call(self, theta_0_prime):
        mu = self.fc_mu(theta_0_prime)
        logvar = self.fc_logvar(theta_0_prime) * 0.3
        eps = tf.random.normal(shape=tf.shape(mu))
        theta_1 = mu + tf.exp(logvar) * eps
        return theta_1, mu, logvar

def compute_kl_loss(mu, logvar):
    logvar = tf.clip_by_value(logvar, -5.0, 2.0)
    return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)

def compute_recon_loss(x, x_recon):
    x = tf.cast(x, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)
    return tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2])

def compute_total_loss(x, x_recon, mu_1, logvar_1, mu_0, logvar_0, ot_loss, rho1=1.0, rho2=1.0, rho3=1.0):
    recon_loss = compute_recon_loss(x, x_recon)
    kl_theta_1 = compute_kl_loss(mu_1, logvar_1)
    kl_theta_0 = compute_kl_loss(mu_0, logvar_0)
    
    total = (
        recon_loss + 
        rho1 * kl_theta_1 + 
        rho2 * kl_theta_0 + 
        rho3 * ot_loss
    )
    return tf.reduce_mean(total)