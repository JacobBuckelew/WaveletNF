import torch
import torch.nn as nn
#from model.NF import BatchNormLayer, CouplingLayer
from model.Wavelet import DiscreteWaveletTransform
from model.Encoder import Encoder

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
                num_entities, 
                st_units,
                mask, 
                wavelet_ =True,
                attention = True,
                b_norm= True, 
                momentum=0.95):
        
        super(WaveletEnhancedCouplingLayer, self).__init__()

        self.hidden_d = hidden_d
        self.wavelet = wavelet
        self.num_entities = num_entities
        self.N = N
        self.wavelet_ = wavelet_
        self.attention_ = attention
        self.num_features = num_features
        # define the primary modules of a wavelet enhanced coupling flow
        if wavelet_:
            self.dwt = DiscreteWaveletTransform(wavelet=wavelet, input_length=N)
            self.output_shape = int(N)
        else:
            self.output_shape = N
        if self.attention_:
            
            self.encoder = Encoder(model_d=self.output_shape, num_features = num_entities,hidden_d=hidden_d, n_heads=n_heads)
            #self.coupling_layer = CouplingLayer(num_entities, mask, st_units, cond_size=num_entities)
        #else:
            #self.coupling_layer = CouplingLayer(num_entities, mask, st_units, cond_size=None)
        #if b_norm:
            #self.batch_norm = BatchNormLayer(num_entities, momentum)
        #else:
            #self.batch_norm = None

    # forward pass performs density estimation
    def forward(self, x):

        if len(x.shape) == 2:
            x = x.reshape(-1, self.num_entities, self.output_shape, 1)

    
        # y is a tuple:  (Approximation, Detail)
        # DWT
        total_log_det = 0
        if self.wavelet_ == True:
            log_det, y = self.dwt(x)
            total_log_det += log_det

        # Attention
        # Transpose detail coefficients to get feature tokens
            D = y[1]
            y = y[0].reshape(-1, self.num_entities)
            #print(D.shape)
        else:
            y = x.reshape(-1, self.num_entities)
            D = x.squeeze()
            #print(D.shape)
        

        if self.attention_ == True:
            #print(D.shape)
            h = self.encoder.compute_attention(D)
            self.attention = h
        # perform RealNVP on Approximation coefficients, conditioning on Detail Coefficient self-attention
            h = h.reshape(-1, self.num_entities)
            self.coupling_layer.attention = h
        else:
            h = None
        
        output, log_det = self.coupling_layer(y, condition=h)
        total_log_det += log_det
        if self.batch_norm:
            output, log_det = self.batch_norm(output)
            total_log_det += log_det

        #print(output.shape)
        if self.attention_ == True:
            return total_log_det, output, self.encoder.A
        else:
            return total_log_det, output

    # inverse pass for sampling, forecasting, etc.
    def inverse(self, x):
        pass

"""
import torch
import torch.nn as nn
from model.NF import BatchNormLayer, CouplingLayer
from model.Wavelet import DiscreteWaveletTransform
from model.Encoder import Encoder


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


class WaveletEnhancedCouplingLayer(nn.Module):
    def __init__(self,
                hidden_d, 
                wavelet,
                N,
                n_heads,
                num_features,
                num_entities, 
                st_units,
                mask, 
                wavelet_ =True,
                attention = True,
                b_norm= True, 
                momentum=0.95):
        
        super(WaveletEnhancedCouplingLayer, self).__init__()

        self.hidden_d = hidden_d
        self.wavelet = wavelet
        self.num_entities = num_entities
        self.N = N
        self.wavelet_ = wavelet_
        self.attention_ = attention
        self.num_features = num_features
        # define the primary modules of a wavelet enhanced coupling flow
        if wavelet_:
            self.dwt = DiscreteWaveletTransform(wavelet=wavelet, input_length=N)
            self.output_shape = int(N)
        else:
            self.output_shape = N
        if self.attention_:
            
            self.encoder = Encoder(model_d=hidden_d, window_length= self.output_shape, num_features = num_entities,hidden_d=hidden_d, n_heads=n_heads)
            self.coupling_layer = CouplingLayer(num_entities, mask, st_units, cond_size=num_entities)
        else:
            self.coupling_layer = CouplingLayer(num_entities, mask, st_units, cond_size=None)
        if b_norm:
            self.batch_norm = BatchNormLayer(num_entities, momentum)
        else:
            self.batch_norm = None

    # forward pass performs density estimation
    def forward(self, x):

        if len(x.shape) == 2:
            x = x.reshape(-1, self.num_entities, self.output_shape, 1)

    
        # y is a tuple:  (Approximation, Detail)
        # DWT
        total_log_det = 0
        if self.wavelet_ == True:
            log_det, y = self.dwt(x)
            total_log_det += log_det

        # Attention
        # Transpose detail coefficients to get feature tokens
            D = y[1]
            y = y[0]
            #y = y[0].reshape(-1, self.num_entities)
            #print(D.shape)
        else:
            y = x.reshape(-1, self.num_entities)
            D = x.squeeze()
            #print(D.shape)
        

        if self.attention_ == True:
            #print(D.shape)
            # compute attention on the detail coefficients
            #h = self.encoder.compute_attention(D)
            # pass the approximation and detail coefficients through cross attention encoder
            h = self.encoder(D, y)
            y = y.reshape(-1, self.num_entities)


            
            

            self.attention = h
        # perform RealNVP on Approximation coefficients, conditioning on Detail Coefficient self-attention
            h = h.reshape(-1, self.num_entities)
            self.coupling_layer.attention = h
        else:
            h = None
        
        output, log_det = self.coupling_layer(y, condition=h)
        total_log_det += log_det
        if self.batch_norm:
            output, log_det = self.batch_norm(output)
            total_log_det += log_det

        #print(output.shape)
        if self.attention_ == True:
            return total_log_det, output, self.encoder.A
        else:
            return total_log_det, output

    # inverse pass for sampling, forecasting, etc.
    def inverse(self, x):
        pass
"""