import torch
import torch.nn as nn
from NF import RealNVP, BatchNormLayer, CouplingLayer
from Wavelet import DiscreteWaveletTransform
from Encoder import Encoder

"""
    WaveletEnhancedCouplingLayer defines the primary model architecture used in our paper.
    - Each Flow "block" consists of Discrete Wavelet Transform --> Cropping --> Attention --> Conditional RealNVP 
    - First block in the sequence will require a zero padding which is handled by DWT
    - WaveletEnhancedCouplingLayers can perform density estimation in the forward direction (anomaly detection) and sampling in the reverse direction(forecasting, data generation)
    
    :param int hidden_d: Hidden dimension size for queries and keys used in self-attention 
    :param str wavelet: family of wavelets used for DWT (e.g. haar)
    :param int N: fixed input sequence length (acts as a max length for determining proper zero padding)
    :param int num_features: input dimensionality 
    :param array st_units: size of hidden layers in the scale and translation networks
    :param int st_layers: number of hidden layers in the scale and translation networks
    :param float momentum: parameter that controls the batch normalization used in RealNVP
    :param int n_heads: number of heads for self-attention computation

"""
class WaveletEnhancedCouplingLayer(nn.Module):
    def __init__(self,
                hidden_d, 
                wavelet,
                N,
                n_heads,
                num_features, 
                st_units,
                mask, 
                b_norm= True, 
                momentum=0.95):
        
        super(WaveletEnhancedCouplingLayer, self).__init__()

        self.hidden_d = hidden_d
        self.wavelet = wavelet
        self.N = N
        self.num_features = num_features
        # define the primary modules of a wavelet enhanced coupling flow
        self.dwt = DiscreteWaveletTransform(wavelet=wavelet, input_length=N)
        self.encoder = Encoder(model_d=num_features, hidden_d=hidden_d, n_heads=n_heads)
        self.coupling_layer = CouplingLayer(num_features, mask, st_units, cond_size=hidden_d)
        
        if b_norm:
            self.batch_norm = BatchNormLayer(num_features, momentum)
        else:
            self.batch_norm = None

    # forward pass performs density estimation
    def forward(self, x):
        # y is a tuple:  (Approximation, Detail)
        # DWT
        total_log_det = 0
        log_det, y = self.dwt(x)
        total_log_det += log_det

        # Attention
        # Transpose detail coefficients to get feature tokens
        D = y[1].T
        h = self.encoder.compute_attention(D)
        self.attention = h

        # perform RealNVP on Approximation coefficients, conditioning on Detail Coefficient self-attention
        log_det, output = self.coupling_layer(y[0], y=h.T)
        total_log_det += log_det

        # Batch Norm
        #TODO Finish the rest of this
        if self.batch_norm:
            log_det, output = self.batch_norm(output)
            total_log_det += log_det

        return total_log_det, output

    # inverse pass for sampling, forecasting, etc.
    def inverse(self, x):
        pass
