import os
from model.WaveletEnhancedCouplingLayer import *
from model.NF import *
from model.Encoder import *
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
                 k,
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
        self.num_entities = num_entities
        self.N = N
        self.attention = attention
        self.wavelet = wavelet
        self.k = k
        self.num_features = num_features
        self.b_norm = b_norm
        self.momentum = momentum
        #st_units = [st_units] * st_layers
        # initialize base distribution
        self.register_buffer('base_mean', torch.zeros(num_entities))
        self.register_buffer('base_var', torch.ones(num_entities))

        # initialize mask
        #print(self.mask.shape)
        #build model
        n = self.N

        # define the encoder used for each wavelet enhanced coupling layer

        if self.wavelet or self.attention:
            self.encoder = Encoder(num_sensors = num_entities, model_d = num_features, window_length = n, hidden_d= hidden_d, n_heads = n_heads, wavelet = wavelet_type,
                                k = k, use_wavelet=wavelet, use_attention=attention)
        else:
            hidden_d = None
            print("running realnvp")

        #self.feature_extractor = nn.Linear(num_features, hidden_d)
        #print("stu units:", st_units)
        #print("st layers:", st_layers)
        self.nf = RealNVP(num_blocks, num_entities, st_units, st_layers, cond_label_size=hidden_d, batch_norm=b_norm)
        #self.flows = nn.ModuleList()
        #self.b_norms = nn.ModuleList() if self.b_norm else None
        
        # Initialize mask
        mask = torch.arange(num_entities).float() % 2
        
        # Build coupling layers
        #for i in range(num_blocks):
            # Add coupling layer
            #self.flows.append(
                #CouplingLayer(num_entities, mask, st_units, cond_size=None)
            #)
            
            # Add batch norm if enabled
            #if self.b_norm:
                #self.b_norms.append(BatchNormLayer(num_entities, momentum))
            
            # Flip mask for next layer
            #mask = 1 - mask

        #self.flow = nn.Sequential(*self.flows)


    # define prior distribution (standard Gaussian)
    @property
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # density estimation
    def forward(self, x, take_mean=True, take_t_mean=True):
        if len(x.shape) ==2:
            x = x.unsqueeze(0)
        #x = self.feature_extractor(x)
        batch, length, num_sensors = x.size()
        #print("Input x shape:", x.shape)
        
        sum_logdet_J = 0

        # encoder step
        if self.attention or self.wavelet:
            H = self.encoder(x)
        #x = self.feature_extractor(x)
        #H = H.reshape(-1, self.num_entities)
            H = H.reshape((-1, H.shape[2]))
        else:
            H = None
        x = x.reshape((-1, num_sensors))
        #print("Encoded H shape:", H.shape)
        #print("Reshaped x shape:", x.shape)
        #for i in range(len(self.flows)):
            #logdet, x = self.flows[i](x, H)
            #sum_logdet_J += logdet
            #if self.b_norm:
                #logdet_bnorm, x = self.b_norms[i](x)
                #sum_logdet_J += logdet_bnorm
        log_prob = self.nf.log_prob(x,H)
        #print("Log prob shape:", log_prob.shape)
        log_prob = log_prob.reshape(batch, -1)
        if take_t_mean == False:
            return log_prob
        else:
            #print("Log prob shape after reshape:", log_prob.shape)
            log_prob = log_prob.mean(dim=1)
            #print(f"Sum logdet: {sum_logdet_J.mean().item():.4f}")
            #print(f"Prior log prob: {self.prior.log_prob(x).sum(1).mean().item():.4f}")
                
            #density = self.log_density(x, sum_logdet_J)
            #print("Density shape before reshape:", density.shape)
            #density = density.reshape(batch, -1)
            #print("Density shape after reshape:", density.shape)
            #density = torch.mean(density, dim=1)
            #print("Density shape after mean:", density.shape)
            #print("log_prob shape:", log_prob.shape)
            if take_mean:
                return torch.mean(log_prob)
            else:
                return log_prob

    def density_t(self, x, take_mean=True):
        size = x.size()
        sum_logdet_J = 0
        if self.attention:
            attention_scores = []
        for layer in self.flow:
            logdet, x, A = layer(x)
            if self.attention:
                attention_scores.append(A)
            sum_logdet_J += logdet
        if self.attention:
            self.attention_scores = attention_scores
        density = self.log_density(x, sum_logdet_J)
        density = density.reshape(size[0], size[2])
        if take_mean:
            return torch.mean(density)
        else:
            return density

    def get_attention(self):
        return self.attention_scores

    # loss function
    def log_density(self, z: torch.Tensor, sum_logdet_J: float):
        # Compute Loss function (log likelihood)
        #print(sum_logdet_J)
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_logdet_J, dim=1)
        return log_likelihood
    
    def get_top_k_features(self):
        return self.encoder.top_k_idx
    
    def get_att_scores(self):
        return self.encoder.att_scores




"""
import os
from model.WaveletEnhancedCouplingLayer import *
import torch.distributions as D

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
        self.attention = attention
        self.num_features = num_features
        self.b_norm = b_norm
        self.momentum = momentum
        self.st_units = [st_units] * st_layers
        # initialize base distribution
        self.register_buffer('base_mean', torch.zeros(num_entities))
        self.register_buffer('base_var', torch.ones(num_entities))

        # initialize mask
        self.mask = mask = torch.arange(num_entities).float() % 2
        #print(self.mask.shape)
        #build model
        self.flows = []
        n = self.N
        for i in range(num_blocks):
            self.flows += [WaveletEnhancedCouplingLayer(hidden_d, 
                                                  wavelet_type, 
                                                  n, 
                                                  n_heads, 
                                                  num_features, 
                                                  num_entities,
                                                  self.st_units, 
                                                  self.mask,
                                                  wavelet,
                                                  attention,
                                                  b_norm, 
                                                  momentum)]
            self.mask = 1 - self.mask

        self.flow = nn.Sequential(*self.flows)


    # define prior distribution (standard Gaussian)
    @property
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # density estimation
    def forward(self, x, take_mean=True):
        if len(x.shape) ==2:
            x = x.unsqueeze(0)
        size = x.size()
        sum_logdet_J = 0
        if self.attention:
            attention_scores = []
        for layer in self.flow:
            if self.attention:
                logdet, x, A = layer(x)
                attention_scores.append(A)
            else:
                logdet, x = layer(x)
            sum_logdet_J += logdet
        if self.attention:
            self.attention_scores = attention_scores
        density = self.log_density(x, sum_logdet_J)
        density = density.reshape(size[0], -1)
        density = torch.mean(density, dim=1)
        if take_mean:
            return torch.mean(density)
        else:
            return density

    def density_t(self, x, take_mean=True):
        size = x.size()
        sum_logdet_J = 0
        if self.attention:
            attention_scores = []
        for layer in self.flow:
            logdet, x, A = layer(x)
            if self.attention:
                attention_scores.append(A)
            sum_logdet_J += logdet
        if self.attention:
            self.attention_scores = attention_scores
        density = self.log_density(x, sum_logdet_J)
        density = density.reshape(size[0], size[2])
        if take_mean:
            return torch.mean(density)
        else:
            return density


    # sampling
    def inverse(self, z, y=None):
        pass

    def get_attention(self):
        return self.attention_scores

    # loss function
    def log_density(self, z: torch.Tensor, sum_logdet_J: float):
        # Compute Loss function (log likelihood)
        #print(sum_logdet_J)
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_logdet_J, dim=1)
        return log_likelihood
"""


