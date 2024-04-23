import os
from model.WaveletEnhancedCouplingLayer import *
import torch.distributions as D
"""
    :param int num_blocks: number of Wavelet-Enhanced Coupling Layer Blocks
    :param float lr: learning rate used for training
    :param
    :param int hidden_d: Hidden dimension size for queries and keys used in self-attention 
    :param str wavelet: family of wavelets used for DWT (e.g. haar)
    :param int N: fixed input sequence length (acts as a max length for determining proper zero padding)
    :param int num_features: input dimensionality 
    :param int st_units: size of hidden layer in the scale and translation networks
    :param int st_layers: number of hidden layers in the scale and translation networks
    :param float momentum: parameter that controls the batch normalization used in RealNVP
    :param int n_heads: number of heads for self-attention computation
"""
class WaveletEnhancedNF(nn.Module):
    def __init__(self, 
                 num_blocks,
                 hidden_d, 
                 wavelet_type,
                 N,
                 n_heads,
                 num_features, 
                 st_units, 
                 st_layers,
                 num_entities,
                 wavelet=True,
                 attention=True,
                 b_norm= True, 
                 momentum=0.95,
                 ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.hidden_d = hidden_d
        self.wavelet_type = wavelet_type
        self.N = N
        self.num_features = num_features
        self.b_norm = b_norm
        self.momentum = momentum
        self.st_units = [st_units] * st_layers
        # initialize base distribution
        self.register_buffer('base_mean', torch.zeros(num_features))
        self.register_buffer('base_var', torch.ones(num_features))

        # initialize mask
        self.mask = mask = torch.arange(num_features).float() % 2
        #print(self.mask.shape)
        #build model
        self.flow = []
        n = self.N
        for i in range(num_blocks):
            self.flow += [WaveletEnhancedCouplingLayer(hidden_d, 
                                                  wavelet_type, 
                                                  n, 
                                                  n_heads, 
                                                  num_features, 
                                                  num_entities,
                                                  self.st_units, 
                                                  self.mask, 
                                                  b_norm, 
                                                  momentum)]
            self.mask = 1 - self.mask

        self.flow = nn.Sequential(*self.flow)


    # define prior distribution (standard Gaussian)
    @property
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # density estimation
    def forward(self, x, take_mean=True):
        size = x.size()
        sum_logdet_J = 0
        for layer in self.flow:
            logdet, x = layer(x)
            sum_logdet_J += logdet
        density = self.log_density(x, sum_logdet_J)
        density = density.reshape(size[0], -1)
        density = torch.mean(density, dim=1)
        if take_mean:
            return torch.mean(density)
        else:
            return density

    # sampling
    def inverse(self, z, y=None):
        pass

    
    # loss function
    def log_density(self, z: torch.Tensor, sum_logdet_J: float):
        # Compute Loss function (log likelihood)
        #print(sum_logdet_J)
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_logdet_J, dim=1)
        return log_likelihood



