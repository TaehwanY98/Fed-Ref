import torch.nn as nn
from pytorch3dunet.unet3d import model

class Custom3DUnet(model.UNet3D):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr', num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs) -> nn.Module:
        super().__init__(in_channels, out_channels, final_sigmoid, f_maps, layer_order, num_groups, num_levels, is_segmentation, conv_padding, conv_upscale, upsample, dropout_prob, **kwargs)
        
        for m in [m for m in self.children()][:-2]:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        for m in [m for m in self.children()][-2:]:
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class Custom2DUnet(model.UNet2D):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr', num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, conv_upscale=2, upsample='default', dropout_prob=0.1, **kwargs) -> nn.Module:
        super().__init__(in_channels, out_channels, final_sigmoid, f_maps, layer_order, num_groups, num_levels, is_segmentation, conv_padding, conv_upscale, upsample, dropout_prob, **kwargs)
        
        for m in [m for m in self.children()][:-2]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        for m in [m for m in self.children()][-2:]:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)