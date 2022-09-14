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


class Planar(nn.Module):
    """
    Planar flow.

        z = f(x) = x + u h(wᵀx + b)

    [Rezende and Mohamed, 2015]
    """
    def __init__(self, dim, nonlinearity=torch.tanh):
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.

        Returns
        -------
        """
        if self.h in (F.elu, F.leaky_relu):
            u = self.u
        elif self.h == torch.tanh:
            scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
            u = self.u + scal * self.w / torch.norm(self.w) ** 2
        else:
            raise NotImplementedError("Non-linearity is not supported.")
        lin = torch.unsqueeze(x @ self.w, 1) + self.b
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z, log_det

    def inverse(self, z):
        raise NotImplementedError("Planar flow has no algebraic inverse.")


class Radial(nn.Module):
    """
    Radial flow.

        z = f(x) = = x + β h(α, r)(z − z0)

    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim):
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(dim):
        init.uniform_(self.z0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det


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


class MAF(nn.Module):
    """
    Masked auto-regressive flow.

    [Papamakarios et al. 2018]
    """
    def __init__(self, dim, hidden_dim = 8, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        for i in range(1, dim):
            self.layers += [base_network(i, 2, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            log_det += alpha
        return x, log_det


class ActNorm(nn.Module):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype = torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype = torch.float))

    def forward(self, x):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma)
        return z, log_det

    def inverse(self, z):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma)
        return x, log_det


class OneByOneConv(nn.Module):
    """
    Invertible 1x1 convolution.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        W, _ = sp.linalg.qr(np.random.randn(dim, dim))
        P, L, U = sp.linalg.lu(W)
        self.P = torch.tensor(P, dtype = torch.float)
        self.L = nn.Parameter(torch.tensor(L, dtype = torch.float))
        self.S = nn.Parameter(torch.tensor(np.diag(U), dtype = torch.float))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype = torch.float),
                              diagonal = 1))
        self.W_inv = None

    def forward(self, x):
        L = torch.tril(self.L, diagonal = -1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal = 1)
        z = x @ self.P @ L @ (U + torch.diag(self.S))
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z):
        if not self.W_inv:
            L = torch.tril(self.L, diagonal = -1) + \
                torch.diag(torch.ones(self.dim))
            U = torch.triu(self.U, diagonal = 1)
            W = self.P @ L @ (U + torch.diag(self.S))
            self.W_inv = torch.inverse(W)
        x = z @ self.W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 32, input_left_bound=None, input_right_bound=None,
                      output_left_bound=1.,output_right_bound=1., hidden_dim = 800, base_network = FCNN, device="cpu"):
        super().__init__()
        self.dim = dim
        self.K = K
        if input_left_bound is None:
            self.input_left_bound=output_left_bound
        if input_right_bound is None:
            self.input_right_bound=output_right_bound
        self.width=(self.input_right_bound-self.input_left_bound)/2
        self.output_left_bound=output_left_bound
        self.output_right_bound=output_right_bound
        self.height=(self.output_right_bound-self.output_left_bound)/2
        self.device = device
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1)).to(self.device)
        for i in range(1, dim):
            self.layers += [base_network(2*i, 3 * K - 1, hidden_dim).to(self.device)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def trig_transform(self,x):
        return torch.cat((torch.cos(torch.tensor(np.pi).to(self.device)*x/self.width),torch.sin(torch.tensor(np.pi).to(self.device)*x/self.width)),axis=-1)

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
            W, H = 2 * self.width * W, 2 * self.height * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, self.input_left_bound, self.input_right_bound,
                      self.output_left_bound,self.output_right_bound,inverse=False)
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
            W, H = 2 * self.width * W, 2 * self.height * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, self.input_left_bound, self.input_right_bound,
                      self.output_left_bound,self.output_right_bound,inverse=True)
            log_det += ld
        return x, log_det


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
