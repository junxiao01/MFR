import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from torchdeq import get_deq
import numpy as np


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(Channel_Attention, self).__init__()

        # feature channel downscale and upscale --> channel weight
        self.convs = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        # y = self.avg_pool(x)
        y = self.convs(x)
        return x * y

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, omega0=10.0, trainable=True):
        super().__init__()
        self.omega_0 = nn.Parameter(omega0*torch.ones(1), requires_grad=trainable)

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0, trainable=True):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_features, in_features)) * 2 - 1, requires_grad=trainable)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_features, )), requires_grad=trainable)
        self.linear = torch.nn.Linear(in_features, out_features)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-torch.pi, torch.pi)

    def forward(self, input):
        norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_features, out_features, bias=True, omega0=10.0, scale0=10.0, trainable=True):
        super().__init__()

        ## omega0 = 10.0
        ## scale0 = 10.0

        self.omega_0 = nn.Parameter(omega0*torch.ones(1), requires_grad=trainable)
        self.scale_0 = nn.Parameter(scale0*torch.ones(1), requires_grad=trainable)


        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale ** 2))


class Filter_Layer(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, filter='real-gabor', trainable=True):
        super().__init__()

        if filter == 'real-gabor':
            self.filter = RealGaborLayer(in_features=in_ch, out_features=in_ch, trainable=trainable)
        elif filter =='sin':
            self.filter = SineLayer(in_features=in_ch, out_features=in_ch, trainable=trainable)
        elif filter == 'gabor':
            self.filter = GaborLayer(in_features=in_ch, out_features=in_ch, trainable=trainable)

        self.linear = nn.Linear(in_features=in_ch, out_features=in_ch, bias=True)

        self.out_linear = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=mid_ch, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=mid_ch, out_features=out_ch, bias=False)
        )

        self.channel_atten = Channel_Attention(channel=in_ch)


    def forward(self, z, x):

        filtered_input = self.filter(x)
        new_z = self.linear(z) * filtered_input
        out = self.out_linear(self.channel_atten(new_z) + x)

        return new_z, out


@register('mfr')
class MFR(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, feats_dim, filter='real-gabor', trainable=True):
        super().__init__()

        if filter == 'real-gabor':
            self.first_filter = RealGaborLayer(in_features=in_dim, out_features=feats_dim, trainable=trainable)
        elif filter =='sin':
            self.first_filter = SineLayer(in_features=in_dim, out_features=feats_dim, trainable=trainable)
        elif filter == 'gabor':
            self.first_filter = GaborLayer(in_features=in_dim, out_features=feats_dim, trainable=trainable)

        self.n_layers = n_layers

        filter_layers = []
        for _ in range(n_layers):
            filter_layers.append(Filter_Layer(in_ch=feats_dim, out_ch=out_dim, mid_ch=feats_dim//2, filter=filter, trainable=trainable))

        self.first_filters = nn.ModuleList(filter_layers)

    def forward(self, x):

        z = self.first_filter(x) + x

        out = 0.0
        for i in range(0, self.n_layers):
          z, out_ = self.first_filters[i](z, x)
          out += out_

        return out