import tensorflow as tf
from keras import Model, layers
from customblocks import FFNBlock

class CriticModel(Model):
    def __init__(self):
        super().__init__()
        self.ffn_block = FFNBlock()
        self.dense2 = layers.Dense(1, activation='linear')

        self.dense3 = layers.Dense(256) #sigmoid tanh?

    def call(self, current_state, goal, action):
        x = tf.concat([current_state, goal, action], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x