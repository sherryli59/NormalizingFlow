import torch
from torch.distributions import MultivariateNormal
from nf.utils import load_position

class EinsteinCrystal:
    def __init__(self, file_dir, dim=3, boxlength=None, alpha=50, device="cpu"):
        super().__init__()
        self.device = device
        self.lattice = load_position(file_dir).reshape(-1,dim).to(self.device)
        self.natoms = list(self.lattice.size())[0]
        #self.variance = (0.5*torch.kthvalue(torch.linalg.norm(self.lattice-self.lattice[0],dim=1),2)[0])**2
        self.alpha = alpha
        self.dim = dim
        self.noise=MultivariateNormal(torch.zeros(self.dim).to(self.device),1/self.alpha* torch.eye(self.dim).to(self.device))
        self.boxlength = boxlength

    def sample(self,nsamples, flatten=True):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            samples=self.lattice+self.noise.sample((nsamples*self.natoms,)).reshape(-1,self.natoms,self.dim).to(self.device)
            if self.boxlength is not None:
                samples -= ((torch.abs(samples) > 0.5*self.boxlength)
                    * torch.sign(samples) * self.boxlength)
            if flatten:
                return samples.reshape(nsamples,-1)
            else:
                return samples
    
    def log_prob(self,x):
        dev_from_lattice=x.reshape(-1,self.natoms,self.dim)-self.lattice
        if self.boxlength is not None:
            dev_from_lattice -= ((torch.abs(dev_from_lattice) > 0.5*self.boxlength)
                * torch.sign(dev_from_lattice) * self.boxlength) 
        return torch.sum(self.noise.log_prob(dev_from_lattice.reshape(-1,self.dim)).reshape(-1,self.natoms),dim=1)
