from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import torch
import numpy as np
import random
from torch.distributions import MultivariateNormal
import sys
sys.path.append("../")
from nf.utils import load_position

class LJ():
    def __init__(self, boxlength, epsilon=1., sigma=1., cutoff=None, shift=True):
        self.boxlength=boxlength
        self.epsilon=epsilon
        self.sigma=sigma
        self.cutoff=cutoff

    def potential(self,particle_pos):
        """
        Calculates Lennard_Jones potential between particles
        Arguments:
        particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        boxlength: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        Returns:
        total_potential: A tensor of shape (n_particles, n_dimensions)
        representing the total potential of the system
        """
        pair_dist = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        to_subtract = ((torch.abs(pair_dist) > 0.5 * self.boxlength)
                    * torch.sign(pair_dist) * self.boxlength)
        pair_dist -= to_subtract
        distances = torch.linalg.norm(pair_dist.float(), axis=-1)
        scaled_distances = distances + (distances == 0)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            if self.shift:
                pow_6_shift = (self.sigma/self.cutoff)**6
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6 - pow_6_shift**2+pow_6_shift)
            else:
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        else:
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        pair_potential = pair_potential *distances_inverse*distances
        total_potential = torch.sum(pair_potential)/2 
        return total_potential   
    
    def force(self,particle_pos):
        """
        Calculates Lennard_Jones force between particles
        Arguments:
            particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        box_length: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        
        Returns:
            total_force_on_particle: A tensor of shape (n_particles, n_dimensions)
        representing the total force on a particle
         """
        pair_dist = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        to_subtract = ((torch.abs(pair_dist) > 0.5 * self.boxlength)
                    * torch.sign(pair_dist) * self.boxlength)
        pair_dist -= to_subtract
        distances = torch.linalg.norm(pair_dist.float(), axis=-1)
        scaled_distances = distances + (distances == 0)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            pair_force = self.epsilon * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*self.sigma*distances_inverse
        else:
            pow_6 = torch.pow(self.sigma/scaled_distances, 6)
            pair_force = self.epsilon * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*self.sigma*distances_inverse
        force_mag = force_mag * distances_inverse
        force = force_mag.unsqueeze(-1) * pair_dist
        total_force = torch.sum(force, dim=1)
        return total_force

class Fe():
    def __init__(self, input_dir, boxlength=None):
        self.lmp=lammps()
        self.lmp.file(input_dir)
        if boxlength is not None:
            boxhi=boxlength/2
            self.lmp.reset_box([-boxhi,-boxhi,-boxhi],[boxhi,boxhi,boxhi],0,0,0)

    def potential(self,pos_dir,traj_len):
        energy=[]
        lmp=self.lmp
        for i in range(traj_len):
            lmp.command("read_dump %s %d x y z box no add yes format xyz"%(pos_dir,i))
            lmp.command("run 0")
            energy.append(lmp.extract_variable("energy"))
        return np.array(energy)
    
    def force(self,pos_dir,traj_len):
        force=[]
        lmp=self.lmp
        for i in range(traj_len):
            lmp.command("read_dump %s %d x y z box no add yes format xyz"%(pos_dir,i))
            lmp.command("run 0")
            force.append(lmp.numpy.extract_fix("force", LMP_STYLE_ATOM, LMP_TYPE_ARRAY))
        return np.array(force)

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

class EinsteinCrystal:
    def __init__(self, file_dir, dim=3, boxlength=None, alpha=50, device="cpu"):
        super().__init__()
        self.device = device
        self.lattice = load_position(file_dir).reshape(-1,dim).to(self.device)
        self.natoms = list(self.lattice.size())[0]
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