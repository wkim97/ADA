from __future__ import print_function
from __future__ import division

import torch
from torch import nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #nn.init.xavier_uniform_(m.weight)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif (classname.find('BatchNorm') != -1):
        if (m.weight is not None) and (m.weight.data is not None):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif (classname.find('Linear') != -1):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()



##############################
#           Generator
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, num_class, normalize=True, kernel=4, stride=2, dropout=0.0):
        super(UNetDown, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride, padding=1, bias=False)
        self.num_class = num_class
        if normalize:
            self.norm = nn.BatchNorm2d(out_size, eps=1e-10)
        else:
            self.norm = None
        self.fn = nn.LeakyReLU(0.2)
        if dropout:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, x, z=None):
        if z is not None:
            width = x.shape[2]
            spatial_tile_z = torch.unsqueeze(torch.unsqueeze(z, -1).expand(-1, -1, width), -1).expand(-1, -1, -1, width)
            out = self.conv( torch.cat((x, spatial_tile_z), 1) )
        else:
            out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)
        out = self.fn(out)
        if self.drop is not None:
            out = self.drop(out)
        return out


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, num_class, output_padding=1, dropout=0.0):
        super(UNetUp, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
        self.num_class = num_class
        self.norm = nn.BatchNorm2d(out_size, eps=1e-10)
        self.fn = nn.ReLU(inplace=True)
        if dropout:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, x, skip_input):
        out = self.upconv(x)
        out = self.norm(out)
        out = self.fn(out)
        if self.drop is not None:
            out = self.drop(out)

        if skip_input is not None:
            out = torch.cat((out, skip_input), 1)
        return out


class AttackGenerator(nn.Module):
    def __init__(self, base_channel_dim, input_img_channel, z_channel, deeper_layer, num_class, last_dim):
        super(AttackGenerator, self).__init__()
        self.deeper_layer = deeper_layer

        self.down0 = UNetDown(input_img_channel + z_channel, base_channel_dim, num_class, kernel=3, stride=2, normalize=False)
        self.down1 = UNetDown(base_channel_dim + z_channel, base_channel_dim, num_class, kernel=3, stride=2)
        self.down2 = UNetDown(base_channel_dim * 1 + z_channel, base_channel_dim * 2, num_class, kernel=3, stride=2,
                              normalize=deeper_layer)
        if deeper_layer:
            self.down3 = UNetDown(base_channel_dim * 2 + z_channel, base_channel_dim * 4, num_class, normalize=False)
            self.up3 = UNetUp(base_channel_dim * 4, base_channel_dim * 2, num_class)
            self.up2 = UNetUp(base_channel_dim * 2 * 2, base_channel_dim * 1, num_class)
        else:
            self.up2 = UNetUp(base_channel_dim * 2, base_channel_dim * 1, num_class, output_padding=0)
            self.up1 = UNetUp(base_channel_dim * 1 * 2, base_channel_dim, num_class)
        self.up0 = UNetUp(base_channel_dim * 2, base_channel_dim, num_class, output_padding=0)

        final = [ nn.Conv2d(base_channel_dim, last_dim, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.Tanh() ]
        self.final = nn.Sequential(*final)

        self.z_encoder = nn.Sequential(
            nn.Linear(z_channel, z_channel),
            nn.ReLU(),
            nn.Linear(z_channel, z_channel),
            nn.ReLU(),
        )

    def forward(self, x, z):
        if z is not None:
            z_encoded = self.z_encoder(z)
        else:
            z_encoded = None
        d0 = self.down0(x, z_encoded)
        d1 = self.down1(d0, z_encoded)
        d2 = self.down2(d1, z_encoded)

        if self.deeper_layer:
            d3 = self.down3(d2, z_encoded)
            u3 = self.up3(d3, d2)
        else:
            u3 = d2

        u2 = self.up2(u3, d1)
        u1 = self.up1(u2, d0)
        u0 = self.up0(u1, None)
        result = self.final(u0)

        return result
