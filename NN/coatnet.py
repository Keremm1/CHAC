from keras import Model
from customblocks import ResidualBlock, STEMBlock, FFNBlock, MBConvBlock, MSHABlock, OutputBlock

class CoAtNetModel(Model):
    def __init__(self, 
                 num_classes, 
                 num_blocks=[2, 3, 5, 2],
                 out_channels=[96, 192, 384, 768],
                 expansion_rate=4, 
                 se_ratio=0.25,
                 sthotastic_depth_rate=0,
                 drop_connect_rate=0,
                 drop_rate=0,
                 block_types=["convulution","convulution","transformer","transformer"],
                 strides=[2,2,2,2],
                 head_dimension=32,
                 stem_filters=64,
                 stem_strides=2,
                 activation='gelu',
                 classifier_activation = 'softmax',
                 use_dw_strides=True
                 ):
                      
        super().__init__()
        
        assert len(num_blocks) == len(out_channels) == len(block_types) == len(strides)        
        
        self.stem_block = STEMBlock(stem_filters, activation=activation, strides=stem_strides)
        
        global_block_id = 0
        self.blocks = []
        total_blocks = sum(num_blocks)
        for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
            stack_stride = strides[stack_id]
            is_conv_block = True if block_type[0].lower() == "c" else False
            if is_conv_block: stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, list) else se_ratio
            for block_id in range(num_block):
                block_stride = stack_stride if block_id == 0 else 1
                block_conv_short_cut = True if block_id == 0 else False
                if is_conv_block: block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, list) else stack_se_ratio
                block_drop_rate = drop_connect_rate * global_block_id / total_blocks
                global_block_id += 1

                if is_conv_block:
                    blocks = (MBConvBlock(
                        expansion_rate, out_channel, block_stride, block_se_ratio, use_dw_strides, activation=activation
                        ),)
                else:
                    blocks = (
                        MSHABlock(
                            out_channel, block_stride, head_dimension
                        ),
                        FFNBlock(
                            expansion_rate, activation=activation
                        )
                    )
                    
                for block in blocks:                       
                    self.blocks.append(ResidualBlock(
                                block, out_channel, block_stride, False if isinstance(block, FFNBlock) else block_conv_short_cut, 
                                block_drop_rate, sthotastic_depth_rate
                    ))

        self.classification_output_block = OutputBlock(num_classes, drop_rate, classifier_activation)

    def call(self, inputs):
            x = self.stem_block(inputs)

            for block in self.blocks:
                x = block(x)

            return self.classification_output_block(x)
