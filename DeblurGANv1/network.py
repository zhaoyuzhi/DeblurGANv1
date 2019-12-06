import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
# Generator contains 2 Auto-Encoders
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T7 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T8 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.T9 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D3 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        
        x = self.E1(x)                                          # out: batch * 64 * 256 * 256
        x = self.E2(x)                                          # out: batch * 128 * 128 * 128
        x = self.E3(x)                                          # out: batch * 256 * 64 * 64
        
        x = self.T1(x)                                          # out: batch * 256 * 64 * 64
        x = self.T2(x)                                          # out: batch * 256 * 64 * 64
        x = self.T3(x)                                          # out: batch * 256 * 64 * 64
        x = self.T4(x)                                          # out: batch * 256 * 64 * 64
        x = self.T5(x)                                          # out: batch * 256 * 64 * 64
        x = self.T6(x)                                          # out: batch * 256 * 64 * 64
        x = self.T7(x)                                          # out: batch * 256 * 64 * 64
        x = self.T8(x)                                          # out: batch * 256 * 64 * 64
        x = self.T9(x)                                          # out: batch * 256 * 64 * 64
        
        x = self.D1(x)                                          # out: batch * 128 * 128 * 128
        x = self.D2(x)                                          # out: batch * 64 * 256 * 256
        x = self.D3(x)                                          # out: batch * out_channel * 256 * 256

        return x

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator for Pix2Pix
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = 'none')
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm)
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 1, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm)
        self.final2 = Conv2dLayer(opt.start_channels * 8, 1, 4, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: grayscale input; img_B: ab embedding output
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 3 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x
