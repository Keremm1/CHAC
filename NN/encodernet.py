from keras import Model
from customblocks import ResidualBlock, STEMBlock, FFNBlock, MBConvBlock, MSHABlock, OutputBlock



class EncoderModel(Model):
    def __init__(self):
        self_encoder_block = 


    def call(self, inputs):
        o_t, self_t, v_t = inputs

