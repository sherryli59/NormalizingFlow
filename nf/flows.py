import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from nf.utils import unconstrained_RQS

# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 800, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim)

    def forward(self, x):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det
'''
class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 32, B = 3, hidden_dim = 800, base_network = FCNN, device="cpu",periodic=True):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.device = device
        self.periodic = periodic
        self.layers = nn.ModuleList()
        for i in range(dim):
            if self.periodic:
                if i == 0:
                    self.layers += [base_network(2, 3 * K - 1, hidden_dim).to(self.device)]
                else:
                    self.layers += [base_network(2*i, 3 * K - 1, hidden_dim).to(self.device)]
            else:
                if i == 0:
                    self.layers += [base_network(1, 3 * K - 1, hidden_dim).to(self.device)]
                else:
                    self.layers += [base_network(i, 3 * K - 1, hidden_dim).to(self.device)]

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def trig_transform(self,x):
        return torch.cat((torch.cos(torch.tensor(np.pi)*x/self.B),torch.sin(torch.tensor(np.pi)*x/self.B)),axis=-1)

    def forward(self, x):
        z = torch.zeros_like(x).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)
        for i in range(self.dim):
            if i == 0:
                input = torch.zeros(x.shape[0],1).to(self.device)
            else:
                input = x[:, :i]
            if self.periodic:
                input = self.trig_transform(input)
            out = self.layers[i](input)
            W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False,tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(x.shape[0]).to(self.device)
        for i in range(self.dim):
            if i == 0:
                input = torch.zeros(z.shape[0],1).to(self.device)
            else:
                input = z[:, :i]
            if self.periodic:
                input = self.trig_transform(input)
            out = self.layers[i](input)
            W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det
'''


class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.
    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 32, B = 3, hidden_dim = 800, base_network = FCNN, device="cpu"):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.device = device
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1)).to(self.device)
        for i in range(1, dim):
            self.layers += [base_network(2*i, 3 * K - 1, hidden_dim).to(self.device)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def trig_transform(self,x):
        return torch.cat((torch.cos(torch.tensor(np.pi)*x/self.B),torch.sin(torch.tensor(np.pi)*x/self.B)),axis=-1)

    def forward(self, x):
        z = torch.zeros_like(x).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](self.trig_transform(x[:, :i]))
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False,tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(x.shape[0]).to(self.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](self.trig_transform(x[:, :i]))
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det
class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Wirnsberger et al. 2020]
    """
    def __init__(self, size, dim=3, K = 32, B = 3, hidden_dim = 800, base_network = FCNN, device="cpu",mask=[1]):
        super().__init__()
        self.size = size
        self.dim = dim
        self.K = K
        self.B = B
        self.device = device
        self.mask = torch.Tensor(mask).long()
        self.unmasked = torch.Tensor([x for x in range(self.dim) if x not in self.mask]).long()
        self.psi = base_network(len(mask)*self.size, (3 * K - 1) * (self.dim-len(self.mask))*self.size, hidden_dim).to(self.device)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0]).to(self.device)
        x=x.reshape(-1,self.size, self.dim)
        lower, upper = x[:, :, self.mask].flatten(start_dim=1), x[:, :, self.unmasked].flatten(start_dim=1)
        out = self.psi(lower).reshape(-1, (self.dim-len(self.mask))*self.size, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower.reshape(-1,self.size,len(self.mask)), upper.reshape(-1,self.size,self.dim-len(self.mask))], dim = 2).flatten(start_dim=1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0]).to(self.device)
        z=z.reshape(-1,self.size, self.dim)
        lower, upper = z[:, :, self.mask].flatten(start_dim=1), z[:, :, self.unmasked].flatten(start_dim=1)
        out = self.psi(lower).reshape(-1, (self.dim-len(self.mask))*self.size, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower.reshape(-1,self.size,len(self.mask)), upper.reshape(-1,self.size,self.dim-len(self.mask))], dim = 2).flatten(start_dim=1), log_det
