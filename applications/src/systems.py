from lammps import lammps, PyLammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import MDAnalysis as MDA
import torch
import numpy as np
import random
from torch.distributions import MultivariateNormal
import sys
sys.path.append("../../")
from src import utils
from ctypes import c_double, POINTER

class LAMMPS():
    def __init__(self,input_dir, save_dir=None,mass=None, init_pos=None, dim=3, temp=1,
        integrator=None, ran_seed=None, cell_len=1):
        self.lmp = lammps()
        self.temp = temp
        if ran_seed == None:
            self.ran_seed = np.random.randint(0,10000)
        else:
            self.ran_seed = ran_seed
        params={"CELL_LEN" : cell_len}
        if save_dir is None:
            save_dir = input_dir
        self.set_input_params(params,input_dir,save_dir+"input.lmp")
        if integrator == "langevin":
            self.lmp.command("fix f all langevin %f %f 0.1 %d zero yes"%(self.temp,self.temp,self.ran_seed))
        self.dim = dim
        self.pylmp = PyLammps(ptr=self.lmp)
        self.nparticles = self.pylmp.system.natoms
        if init_pos is not None:
           self.set_position(init_pos)
        if mass == None:
            self.mass = np.ones(self.nparticles)
        else:
            self.mass = np.broadcast_to(np.array(mass),self.nparticles)
  
            
    def set_input_params(self, params, template, input):
        with open(template, "r") as template:
            with open (input,"w") as output:
                for line in template:
                    output.write(line.format(**params))
        self.lmp.file(input)


    def command(self, str):
        self.lmp.command(str)

    def get_potential(self,position=None):
        if position is not None:
            self.set_position(position)
        self.pylmp.run(0)
        return self.lmp.numpy.extract_variable("energy")

    def get_position(self):
        return np.array(np.ctypeslib.as_array(self.lmp.gather_atoms("x",1,3)))

        #return np.array(np.ctypeslib.as_array(self.lmp.gather_atoms("x",1,3)))
        #pos_unsorted=self.lmp.numpy.extract_atom("x")
        #id = self.lmp.numpy.extract_atom("id")
        #position = np.array([x for _, x in sorted(zip(id, pos_unsorted))])
        #print(position)
        #return position
        
    def set_position_old(self,position):
        if isinstance(position,str):
            position = utils.load_position(position).flatten()
        else:
            position = position.flatten()
        position = position.ctypes.data_as(POINTER(c_double))
        self.lmp.scatter_atoms("x",1,3, position)

    def set_position(self,position):
        if isinstance(position,str):
            position = utils.load_position(position).reshape(-1,self.dim)
        else:
            position = position.reshape(-1,self.dim)
        id = self.lmp.numpy.extract_atom("id")
        for i,index in zip(range(self.nparticles), id):
            self.pylmp.atoms[i].position= position[index-1]

    def set_velocity(self,velocity):
        velocity = velocity.reshape(-1,self.dim)
        id = self.lmp.numpy.extract_atom("id")
        for i,index in zip(range(self.nparticles), id):
            self.pylmp.atoms[i].velocity= velocity[index-1]


    def set_velocity_old(self,velocity):
        velocity = velocity.flatten()
        velocity = velocity.ctypes.data_as(POINTER(c_double))
        self.lmp.scatter_atoms("v",1,3, velocity)

    def integration_step(self,path_len=1,dt=None, init_pos=None ):
        if init_pos is not None:
            self.set_position(init_pos)
        #print("velocity",self.pylmp.atoms[0].velocity)
        if dt is not None:
            self.pylmp.command("timestep %f"%dt)
        self.pylmp.run(int(path_len))
        position = self.get_position()
        potential = self.lmp.numpy.extract_variable("energy")
        return position, potential


        
class SimData():
    def __init__(self, pos_dir=None, device="cpu",data_type="xyz"):
        self.device = device
        self.data_type = data_type
        if pos_dir is not None:
            if data_type == "xyz":
                traj = MDA.coordinates.XYZ.XYZReader(pos_dir)
                self.traj = torch.tensor(np.array([np.array(traj[i]) for i in range(len(traj))]),requires_grad=True).to(device)
            elif data_type == "pt":
                self.traj = torch.tensor(torch.load(pos_dir)).float().to(device)
            elif data_type == "npy":
                self.traj = torch.tensor(np.load(pos_dir)).float().to(device)
                self.traj = self.traj.reshape(len(self.traj),-1)
    
    def sample(self,nsamples, flatten=True, random=True):
        samples = utils.subsample(self.traj,nsamples, self.device, random=random)
        if flatten:
            return samples.reshape(nsamples,-1)
        else:
            return samples
    def update_data(self,file,append=False):
        traj = self.load_traj(file)
        if append:
            self.traj = torch.cat((self.traj,traj),axis=0)
        else:
            self.traj = traj
    def load_traj(self,pos_dir):
        if self.data_type == "xyz":
            traj = MDA.coordinates.XYZ.XYZReader(pos_dir)
            traj = torch.tensor(np.array([np.array(traj[i]) for i in range(len(traj))]),requires_grad=True).to(self.device)
        elif self.data_type == "pt":
            traj = torch.tensor(torch.load(pos_dir)).float()
        elif self.data_type == "npy":
            traj = torch.tensor(np.load(pos_dir)).float()
            traj = traj.reshape(len(self.traj),-1)
        return traj
    
class LJ(SimData):
    def __init__(self, pos_dir=None, boxlength=None, device="cpu", epsilon=1., sigma=1., cutoff=None, shift=True):
        super().__init__(pos_dir,device)
        self.epsilon=epsilon
        self.sigma=sigma
        self.cutoff=cutoff
        self.shift=shift
        self.boxlength=boxlength


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
        total_potential = torch.sum(pair_potential,axis=(-1,-2))/2 
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

class Fe(LAMMPS,SimData):
    def __init__(self, input_dir, save_dir="./",mass=None, init_pos=None, dim=3, temp=1,
        integrator=None, ran_seed=None, cell_len=1, pos_dir=None, device="cpu"):
        LAMMPS.__init__(self,input_dir, save_dir, mass, init_pos, dim, temp,
        integrator, ran_seed, cell_len)
        SimData.__init__(self,pos_dir, device)

    def potential(self,pos,traj_len=None):
        energy=[]
        if isinstance(pos,str): #interpret it as position directory
            lmp=self.lmp
            for i in range(traj_len):
                lmp.command("read_dump %s %d x y z box no add yes format xyz"%(pos,i))
                lmp.command("run 0")
                energy.append(lmp.extract_variable("energy"))
        else: # interpret as a trajectory with dimension (nframes,nparticles,ndims)
            pos=pos.detach().cpu().numpy()
            for i in range(len(pos)):
                energy.append(self.get_potential(pos[i]))
        return np.array(energy)
    
    def force(self,pos_dir,traj_len):
        force=[]
        lmp=self.lmp
        for i in range(traj_len):
            lmp.command("read_dump %s %d x y z box no add yes format xyz"%(pos_dir,i))
            lmp.command("run 0")
            force.append(lmp.numpy.extract_fix("force", LMP_STYLE_ATOM, LMP_TYPE_ARRAY))
        return np.array(force)



class GaussianMixture:
    def __init__(self, centers, vars, npoints=None,dim=3,device="cpu"):
        self.dim = dim
        self.device = device
        if isinstance(centers,str):
            self.centers = utils.load_position(centers).reshape(-1,dim).to(self.device)
        else:
            self.centers=torch.tensor(centers).float().to(self.device)    
        self.ncenters=len(self.centers)
        self.vars = torch.tensor(vars).float().to(self.device)
        if self.vars.dim()==0:
            self.vars = self.vars.expand(self.ncenters)
        if npoints == None:
            self.nparticles = self.ncenters
        else:
            self.nparticles = npoints
        self.dist=[]
        for i in range(self.ncenters):
            self.dist.append(MultivariateNormal(self.centers[i], self.vars[i]*torch.eye(self.dim).to(self.device)))
    def sample(self,nsamples,flatten=True):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            which_dist=torch.tensor([random.randint(0,self.ncenters-1) for _ in range(nsamples*self.nparticles)])
            samples = torch.stack([self.dist[which_dist[i]].sample((1,)) for i in range(nsamples*self.nparticles)])
            if flatten:
                return samples.reshape((nsamples,-1))
            else:
                return samples.reshape((nsamples,self.nparticles,self.dim))

    def log_prob(self,x):
        x=x.reshape(-1,self.dim)
        prob=0
        for i in range(self.ncenters):
            prob+=1/self.ncenters*torch.exp(self.dist[i].log_prob(x))
        return torch.sum(torch.log(prob).reshape(-1,self.nparticles),axis=1)

    def potential(self,x):
        return -self.log_prob(x)
    
    def get_potential(self,x=None):
        if x is not None:
            return self.potential(x)
        else:
            return self.potential(self.position)

    def force(self,x):
        x.requires_grad_()
        pot=self.potential(x)
        return -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]

    def get_force(self,x):
        x.requires_grad_()
        pot=self.potential(x)
        return -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]
    
    def set_position(self,position):
        self.position = position.flatten()


    def get_position(self):
        return self.position

    def set_velocity(self,velocity):
        self.velocity = velocity.flatten()

    def integration_step(self,path_len=1,dt=0.005, init_pos=None, init_velocity=None):
        if init_pos is not None:
            self.set_position(init_pos)
        if init_velocity is not None:
            self.set_velocity(init_velocity)
        self.force = self.get_force(self.position)
        if dt is None:
            dt=0.005
        for _ in range(path_len):
            new_position = self.position + self.velocity*dt+ self.force/2*(dt**2)
            new_force = self.get_force(self.position)
            self.velocity = self.velocity + dt*(self.force + new_force)/2
            self.force = new_force
            self.position = new_position
        potential = self.potential(self.position)
        return self.position, potential

class EinsteinCrystal:
    def __init__(self, centers, dim=3, boxlength=None, alpha=50, device="cpu"):
        self.device = device
        if isinstance(centers,str):
            self.centers = utils.load_position(centers).reshape(-1,dim).to(self.device)
        else:
            self.centers=torch.tensor(centers).float().to(self.device) 
        self.natoms = self.centers.shape[0]
        self.alpha = alpha
        self.dim = dim
        self.noise=MultivariateNormal(torch.zeros(self.dim).to(self.device),1/self.alpha* torch.eye(self.dim).to(self.device))
        self.boxlength = boxlength

    def sample(self,nsamples, flatten=True):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            samples=self.centers+self.noise.sample((nsamples*self.natoms,)).reshape(-1,self.natoms,self.dim).to(self.device)
            if self.boxlength is not None:
                samples -= ((torch.abs(samples) > 0.5*self.boxlength)
                    * torch.sign(samples) * self.boxlength)
            if flatten:
                return samples.reshape(nsamples,-1)
            else:
                return samples
    
    def log_prob(self,x):
        dev_from_lattice=x.reshape(-1,self.natoms,self.dim)-self.centers
        #return -0.5*self.alpha*torch.linalg.norm(dev_from_lattice,dim=(1,2))**2
        if self.boxlength is not None:
            dev_from_lattice -= ((torch.abs(dev_from_lattice) > 0.5*self.boxlength)
                * torch.sign(dev_from_lattice) * self.boxlength) 
        return torch.sum(self.noise.log_prob(dev_from_lattice.reshape(-1,self.dim)).reshape(-1,self.natoms),dim=1)
    
    def potential(self,x):
        return -self.log_prob(x)
    
    def get_force(self,x):
        x.requires_grad_()
        pot=self.potential(x)
        return -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]
    
    def set_position(self,position):
        self.position = position.flatten()

    def set_velocity(self,velocity):
        self.velocity = velocity.flatten()

    def integration_step(self,path_len=1,dt=0.005, init_pos=None, init_velocity=None):
        if init_pos is not None:
            self.set_position(init_pos)
        if init_velocity is not None:
            self.set_velocity(init_velocity)
        self.force = self.get_force(self.position)
        for _ in range(path_len):
            new_position = self.position + self.velocity*dt+ self.force/2*(dt**2)
            new_force = self.get_force(self.position)
            self.velocity = self.velocity + dt*(self.force + new_force)/2
            self.force = new_force
            self.position = new_position
        potential = self.potential(self.position)
        return self.position, potential