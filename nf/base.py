import torch
from torch.distributions import MultivariateNormal
from nf.utils import load_position
import random

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

class GaussianMixture():
    def __init__(self, centers, vars, npoints,dim=3):
        self.dim=dim
        self.ncenters=len(centers)
        self.centers=torch.tensor(centers).float()
        self.vars=torch.tensor(vars).float()
        self.npoints=npoints
        self.dist=[]
        for i in range(self.ncenters):
            self.dist.append(MultivariateNormal(self.centers[i], self.vars[i]*torch.eye(self.dim)))

    def sample(self,nsamples,flatten=True):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            which_dist=torch.tensor([random.randint(0,self.ncenters-1) for _ in range(nsamples*self.npoints)])
            samples = torch.stack([self.dist[which_dist[i]].sample((1,)) for i in range(nsamples*self.npoints)])
            if flatten:
                return samples.reshape((nsamples,-1))
            else:
                return samples.reshape((nsamples,self.npoints,self.dim))

    def log_prob(self,x):
        x=x.reshape(-1,self.dim)
        prob=0
        for i in range(self.ncenters):
            prob+=1/self.ncenters*torch.exp(self.dist[i].log_prob(x))
        return torch.sum(torch.log(prob).reshape(-1,self.npoints),axis=1)

    def potential(self,x):
        return -self.log_prob(x)
    
    def force(self,x):
        x.requires_grad_()
        pot=self.potential(x)
        return -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]