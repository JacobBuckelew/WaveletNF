import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math


#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math
import copy


# --------------------
# Model layers and helpers
# --------------------

def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_size>1:
        if input_order == 'sequential':
            degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [torch.arange(hidden_size) % (input_size - 1)]
            degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

        elif input_order == 'random':
            degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
                degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]
    else:
        degrees += [torch.zeros([1]).long()]
        for _ in range(n_hidden+1):
            degrees += [torch.zeros([hidden_size]).long()]
        degrees += [torch.zeros([input_size]).long()]
    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]

#%%

def create_masks_pmu(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
    masks[0] = masks[0].repeat_interleave(3, dim=1)
    masks[-1] = masks[-1].repeat_interleave(3, dim=0)

    return masks, degrees[0]
#%%
class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:,i] = u[:,i] * torch.exp(loga[:,i]) + m[:,i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MADE_Full(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks_pmu(int(input_size/3), hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class MAF_Full(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE_Full(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)



class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

















"""
class BatchNormLayer(nn.Module):

    def __init__(self, num_features, momentum=.95, eps=1e-5):
        super(BatchNormLayer, self).__init__()
        self.momentum = momentum
        self.eps = eps

        # Beta and Gamma will be model parameters
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))

        # Define non-trainable running averages of variance and mean
        # we can update the in-place
        self.register_buffer('run_m', torch.zeros(num_features))
        self.register_buffer('run_v', torch.ones(num_features))

 
    def forward(self, x, cond=None):
        # original minibatch algorithm

        if self.training:
            # calculate mean and variance of minibatch
            self.b_mean = x.mean(0)
            self.b_var = x.var(0)

            # update running mean and variance in-place (Dinh)
            self.run_m.mul_(self.momentum).add_((1 - self.momentum) * self.b_mean.data)
            self.run_v.mul_(self.momentum).add_((1- self.momentum) * self.b_var.data)
            var = self.b_var
            mean = self.b_mean
        else:
            mean, var = self.run_m, self.run_v
        
        # Normalize input and scale + shift using gamma and beta

        y = self.gamma.exp() * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta

        # Compute log determinant of jacobian (Dinh)
        log_det = self.gamma - (0.5 * torch.log(self.eps + var))
        return y, log_det.expand_as(x)
    
    def inverse(self, y, cond=None):
        if self.training:
            mean = self.b_mean
            variance = self.b_var
        else:
            mean = self.run_m
            variance = self.run_v

        x = (y - self.beta) * torch.exp(-self.gamma) * torch.sqrt(variance + self.eps) + mean

        log_det = 0.5 * (torch.log(variance + self.eps) - self.gamma)

        return x, log_det.expand_as(x)

class CouplingLayer(nn.Module):
    def __init__(self, n_features, mask, st_units, cond_size=None):
        super(CouplingLayer, self).__init__()
        self.register_buffer('mask', mask)
        self.snet = []
        self.tnet = []
        self.attention = None

        # build the snet (scaling function)
        if cond_size:
            self.snet.append(nn.Linear(n_features + cond_size, st_units[0]))
        else:
            self.snet.append(nn.Linear(n_features, st_units[0]))
        for units in st_units:
            self.snet.append(nn.Tanh())
            self.snet.append(nn.Linear(units, units))
        self.snet.append(nn.Tanh())
        self.snet.append(nn.Linear(st_units[0], n_features))
        self.snet = nn.Sequential(*self.snet)

        # build tnet
        if cond_size:
            self.tnet.append(nn.Linear(n_features + cond_size, st_units[0]))
        else:
            self.tnet.append(nn.Linear(n_features, st_units[0]))
        for units in st_units:
            self.tnet.append(nn.ReLU())
            self.tnet.append(nn.Linear(units, units))
        self.tnet.append(nn.ReLU())
        self.tnet.append(nn.Linear(st_units[0], n_features))
        self.tnet = nn.Sequential(*self.tnet)

    def conditioner(self, x, condition):
        #if self.attention is not None:
            #x = self.attention * x
        if condition is not None:
            x = torch.concat((x, condition), axis= -1)
        #print(x.shape)
        # perform affine coupling on x_[1:d]
        # scale function
        scale_x = x
        
        for net in self.snet:
            scale_x = net(scale_x)
        scale = scale_x
        # translation function
        translate_x = x
        for net in self.tnet:
            translate_x = net(translate_x)
        shift = translate_x
        return scale, shift
    
    # Perform density estimation: real-world dist. ---> base dist.
    def forward(self, x, condition):
        x_mask = x * self.mask
        reverse_mask = 1 - self.mask
        #print("mask:", reverse_mask.shape)
        # use our neural networks for approximating s() and t()
        scale, translate = self.conditioner(x_mask, condition)
        # RealNVP using Masking to partition the data
        y = x_mask + (reverse_mask * ((x - translate) * torch.exp(-1 * scale)))

        # update the log absolute det of Jacobian
        det_jacobian = -1 * reverse_mask * scale 
        #print("jacobian:", det_jacobian.shape)
        return y, det_jacobian

    # Sampling: base dist. --> real-world dist.
    def inverse(self, y, conditions):
        y_mask = y * self.mask
        reverse_mask = 1 - self.mask
        scale, translate = self.conditioner(y_mask, conditions)
        x = y_mask + (reverse_mask * (y * torch.exp(scale) + translate))

        det_jacobian = reverse_mask * scale
        return x, det_jacobian


class RealNVP(nn.Module):
    def __init__(self, 
                 num_features, 
                 st_units, 
                 st_layers, 
                 cond_size,
                 mask, 
                 b_norm= True, 
                 momentum=0.95):
        
        super().__init__()

        # define a prior (base) distribution ~ Standard Normal
        self.register_buffer('base_mean', torch.zeros(num_features))
        self.register_buffer('base_var', torch.ones(num_features))
        self.momentum = momentum
        self.b_norm = b_norm
        self.cond_size = cond_size
        self.st_units = [st_units] * st_layers

        # define a mask

        #self.mask = torch.arange(num_features).float() % 2

        # build model
        self.flow = []
    
        # append coupling layers
        self.flow += [CouplingLayer(n_features = num_features, mask = self.mask, st_units =self.st_units, cond_size=self.cond_size)]
        # batch norm
        if self.b_norm:
            self.flow += [BatchNormLayer(num_features =num_features, momentum=momentum)]
        # define as parameter for optimization
        self.flow = nn.Sequential(*self.flow)

    @property        
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # Inference Direction
    def forward(self, x, y=None):
        # sum up log det of jacobian as data passes through layers
        sum_ldj = 0
        for module in self.flow:
            x, ldj = module(x,y)
            sum_ldj = sum_ldj + ldj
        return x, sum_ldj
    
    # Sampling direction
    def inverse(self, z, y=None):
        sum_ldj = 0
        for module in self.flow[::-1]:
            z, ldj = module(z,y)
            sum_ldj = sum_ldj + ldj
        return z, sum_ldj
    
    # Loss function
    def log_density(self, x, y=None):
        z, sum_ldj = self.forward(x, y)
        #print("z:", z)
        # Log Likelihood formula using change of variables technique
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_ldj, dim=1)
        #log_likelihood = self.prior.log_prob(z) + sum_ldj
        return log_likelihood
"""