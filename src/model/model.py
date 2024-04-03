from WaveletEnhancedCouplingLayer import *
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
                 lr,
                 max_epochs,
                 hidden_d, 
                 wavelet,
                 N,
                 n_heads,
                 num_features, 
                 st_units, 
                 st_layers,
                 b_norm= True, 
                 momentum=0.95):

        
        self.num_blocks = num_blocks
        self.lr = lr
        self.hidden_d = hidden_d
        self.max_epochs = max_epochs
        self.wavelet = wavelet
        self.N = N
        self.num_features = num_features
        self.b_norm = b_norm
        self.momentum = momentum
        self.st_units = [st_units] * st_layers
        # initialize base distribution
        self.register_buffer('base_mean', torch.zeros(num_features))
        self.register_buffer('base_var', torch.ones(num_features))

        # initialize mask
        self.mask = [1 for x in range(num_features)] + [0 for x in range(hidden_d)]

        #build model
        self.flow = []
        for i in range(num_blocks):
            self.flow += [WaveletEnhancedCouplingLayer(hidden_d, 
                                                  wavelet, 
                                                  N, 
                                                  n_heads, 
                                                  num_features, 
                                                  self.st_units, 
                                                  self.mask, 
                                                  b_norm, 
                                                  momentum)]
            if self.b_norm:
                self.flow += [BatchNormLayer(num_features, momentum)]
            self.mask = 1 - self.mask

        self.flow = nn.Sequential(*self.flow)


    # define prior distribution (standard Gaussian)
    @property
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # density estimation
    def forward(self, x):
        sum_logdet_J = 0
        for layer in self.flow:
            logdet, x = layer(x)
            sum_logdet_J += logdet
        return x, sum_logdet_J

    # sampling
    def inverse(self, z, y=None):
        pass

    
    # loss function
    def log_density(self, x: torch.Tensor):
        z, sum_logdet_J = self.forward(x)
        # Compute Loss function (log likelihood)
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_logdet_J, dim=1)
        return log_likelihood

    def train(self):
        pass

    def validate(self):
        pass


