from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import torch
from torch.distributions import MultivariateNormal
from . import utils as util
import numpy as np
import math

class HMC:
    def __init__(self, simulation, init_pos=None, path_len=1, dt=None, mass=None, dim=3, beta=1.0, init_beta=None):
        self.simulation = simulation
        self.dt = dt
        self.path_len = path_len
        self.beta = beta
        self.dim = dim
        self.nparticles = simulation.nparticles
        self.position = simulation.get_position()
        self.potential = simulation.get_potential()
        if init_pos is not None:
            self.simulation.set_position(init_pos)
        if mass == None:
            self. mass = torch.ones(self.nparticles)
        else:
            self. mass = mass
        mass_inverse = (1/self.mass).expand(self.dim,self.nparticles).transpose(0,1).flatten()
        if init_beta is None:
            init_beta = beta
        self.v_dist = MultivariateNormal(torch.zeros(self.dim*self.nparticles),torch.diag(mass_inverse)/init_beta)
  
    def generate_v(self):
        v = self.v_dist.sample((1,))
        log_prob = self.v_dist.log_prob(v)
        return v.flatten(),log_prob
        
    def run_sim(self, v=None):
        if v is None:
            v, log_prob = self.generate_v()
        else:
            log_prob = self.v_dist.log_prob(v)
        self.simulation.set_velocity(v)
        position,potential = self.simulation.integration_step(self.path_len,self.dt)
        return position, potential, log_prob
    
    def hmc(self, epochs=1,init_pos=None):
        position_list=[]
        potential_list=[]
        naccept = 0
        if init_pos is not None:
            self.simulation.set_position(init_pos)
            self.position = init_pos
            self.potential = self.simulation.get_potential()
            print(self.potential)
        for i in range(epochs):
            position_list.append(torch.tensor(self.position).float().flatten())
            potential_list.append(self.potential)
            position,potential,log_prob = self.run_sim()
            acc_prob = math.exp((self.potential-potential)*self.beta)
            print("instantaneous acc prob:", acc_prob)
            if torch.rand(1)<acc_prob:
                self.position =position.flatten()
                self.potential = potential
                naccept +=1
            else:
                self.simulation.set_position(self.position)

        return torch.stack(position_list), torch.tensor(potential_list), torch.tensor(log_prob), naccept/epochs



