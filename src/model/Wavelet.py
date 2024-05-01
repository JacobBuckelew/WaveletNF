#import pywt
import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
import ptwt, pywt

"""
    DiscreteWaveletTransform implements DWT, zero padding, and cropping using the PyWavelets library (https://pywavelets.readthedocs.io/en/latest/)

    :param str wavelet: wavelet family used for DWT (see documentation for options)
    :param int input_length: a fixed value (power of 2) for the length of the input sequence. Used for determining the amount of padding if length of the actual input sequence < input_length
    :param int level: How many decompositions to be performed. By default, this is 1 since each Wavelet-Enhanced Coupling Layer performs 1 level of decomposition
"""

class DiscreteWaveletTransform(nn.Module):

    def __init__(self, wavelet: str, input_length: int):
        super(DiscreteWaveletTransform, self).__init__()

        self.wavelet = wavelet
        self.input_length = input_length
        self.output_length = math.ceil(input_length / 2)

    def zero_padding(self, x: torch.Tensor) -> torch.Tensor:
        # inputs are of shape (Dimension, Seq_Length). Pad zeros to the last dimension
        device = x.device
        padding = torch.zeros(x.shape[0], x.shape[1], self.input_length - x.shape[-2], x.shape[-1]).to(device)
        output = torch.cat((x, padding), -2)
        return output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad input before transform
        device = x.device
        #print(x.shape)
        if x.shape[-2] < self.input_length:
            x = self.zero_padding(x)
        x = x.cpu()
        x = x.squeeze()
        #x = x.to(device)
        # calculate Discrete Wavelet Transform
        #self.forward_kernel = self.forward_kernel.to(device)
        y = ptwt.wavedec(x, wavelet=pywt.Wavelet(self.wavelet), level=1)
        A = y[0].to(device)
        D = y[1].to(device)
        #A, D = self.forward_kernel(x)
        #D = D[0][:, :, 1, :, :]
        #print(D[0, 0, :, :])
        # Assume we are using an Orthonormal Wavelet family (e.g. haar). Then, Determinant of DWT is 1
        log_det = 0

        return log_det, (A,D)

    def inverse(self):
        pass