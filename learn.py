import pdb
import warnings
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from dpftrl.ftrl_noise import CummuNoiseEffTorch
from dpftrl.optimizers import FTRLOptimizer


def log_laplace_integral(offset, radius):
    with torch.no_grad():
        mask = offset > radius
    return torch.where(mask,
                       offset - torch.log(torch.sinh(radius)),
                       radius - torch.log(torch.exp(radius) - torch.cosh((~mask).type(offset.dtype)*offset)))

def censored_l3(a, b, theta, phi):
    return log_laplace_integral(torch.absolute(theta - .5 * phi * (a + b)),
                                .5 * phi * (b - a))


class ReparameterizedLinear(nn.Module):

    def __init__(self, n_features, n_quantiles, theta=None, phi=1.):
        super().__init__()
        if n_features:
            self.linear = nn.Linear(n_features, n_quantiles)
            if not theta is None:
                self.linear.bias = nn.Parameter(theta * torch.ones(n_quantiles))
        else:
            self.linear = nn.Linear(1, n_quantiles, bias=False)
            self.linear.weight = nn.Parameter(theta * torch.ones(n_quantiles, 1))
        self.phi = nn.Linear(1, n_quantiles, bias=False)
        self.phi.weight = nn.Parameter(phi * torch.ones(n_quantiles, 1))
        self.ones = None

    def forward(self, f):
        if self.ones is None:
            shape = [len(f), 1] if len(f.shape) == 2 else [1]
            self.ones = torch.ones(shape, device=f.device)
        return self.linear(self.ones if self.linear.bias is None else f), self.phi(self.ones)

    def predict(self, f):
        with torch.no_grad():
            if self.linear.bias is None:
                return self.linear.weight[:,0] / self.phi.weight[:,0], 1. / self.phi.weight[:,0]
            return self.linear(f) / self.phi.weight[:,0], 1. / self.phi.weight[:,0]

    def project(self, lower):
        with torch.no_grad():
            self.phi.weight.data = self.phi.weight.data.clamp(lower)


class DPFTRL:
    def __init__(self, model, lr, epsilon, delta, n, batch, clip):
        self.optimizer = FTRLOptimizer(model.parameters(), 0., False)
        self.lr = lr
        self.multiplier = np.sqrt(2.*np.ceil(np.log2(n//batch+1)) * np.log(1./delta)) / epsilon / batch
        self.noise = CummuNoiseEffTorch(self.multiplier * clip, [p.shape for p in model.parameters()], 'cpu')
        self.delta = delta

    def step(self):
        self.optimizer.step((1./self.lr, self.noise()))

    def zero_grad(self):
        self.optimizer.zero_grad()
