import pywt
import math
import torch
import torch.nn as nn

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
        output = torch.cat((x, torch.zeros(x.shape[-2], self.input_length - x.shape[-1])), -1)
        return output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad input before transform
        
        if x.shape[-1] < self.input_length:
            x = self.zero_padding(x)
        x = x.numpy()
        # calculate Discrete Wavelet Transform
        # Loop over each level of decomposition (level = 1 for each RealNVP Block)
        A, D = pywt.dwt(x, wavelet=self.wavelet)
        # Assume we are using an Orthonormal Wavelet family (e.g. haar). Then, Determinant of DWT is 1
        log_det = 0
        #self.A = 

        return log_det, (torch.Tensor(A),torch.Tensor(D))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.output_length, f"Input sequence length is {x.shape[-1]}, but should be {self.output_length} for computing the inverse DWT"



if __name__ == "__main__":
    DWT = DiscreteWaveletTransform(wavelet="haar", input_length=64, level=4)
    x = torch.rand((10,50))
    log_det, y = DWT(x)
    print(y[0])
    print(y[1].shape)



    

    



