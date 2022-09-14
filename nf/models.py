import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows, device="cpu"):
        super().__init__()
        self.device=device
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]: 
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, log_det = self.inverse(z)
        log_px=self.prior.log_prob(z)-log_det
        return x.data,log_px.data,z.data

    def evaluate(self,x):
        z, prior_logprob, log_det = self.forward(x)
        log_px=prior_logprob+log_det
        return log_px.data
