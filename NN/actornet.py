import tensorflow as tf
from keras import Model, layers
import tensorflow_probability as tfp
from customblocks import FFNBlock

class SthotasticActorModel(Model):
    def __init__(self):
        super().__init__()
        self.ffn_block = FFNBlock()
        self.dense_mu = layers.Dense(1, activation='linear')
        self.dense_std = layers.Dense(1, activation='softplus')

    def call(self, current_state, goal):

        x = tf.concat([current_state, goal], axis=1)
        x = self.dense(x)
        mu = self.dense_mu(x)
        std = self.dense_std(x)
        z = tfp.distributions.Normal(loc=0, scale=1).sample()
        action = mu + std * z
        return action
        
