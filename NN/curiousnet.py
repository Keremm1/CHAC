import tensorflow as tf
from keras import Model
from customblocks import FFNBlock

class CuriousModel(Model):
    def __init__(self):
        super().__init__()

        self.ffn_block = FFNBlock()

    def call(self, current_state, action):
        
        x = self.ffn_block (tf.concat(current_state, action))
        x = self.ffn_block (x)
        predicted_state = self.ffn_block (x)

        return predicted_state
