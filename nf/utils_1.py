import numpy as np
import torch
import torch.nn.functional as F
import MDAnalysis as MDA
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import MDAnalysis as MDA
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
import sys
sys.path.append("../")
from applications.config import get_cfg_defaults
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY

"""
Implementation of rational-quadratic splines in this file is taken from
https://github.com/bayesiains/nsf.

"""

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def load_position(dir):
    traj = MDA.coordinates.XYZ.XYZReader(dir)
    pos = torch.from_numpy(np.array([np.array(traj[i]) for i in range(len(traj))])).flatten(start_dim=1)
    return pos

def read_coord(dir,format="torch"):
    with open(dir, 'rb') as coord:
        n_atoms=int(coord.readline())
        counter=0
        coord.seek(0)
        pos=[]
        while True:
            line = coord.readline()
            if not line:
                break
            if (counter%(n_atoms+2)==0):
                pos.append(np.zeros((n_atoms,3))) 
            if (counter%(n_atoms+2)>1): 
                pos[-1][counter%(n_atoms+2)-2]=line.split()[1:4]
            counter+=1
        if format=="torch":
            pos=torch.from_numpy(np.array(pos))
        else:
            pos=np.array(pos)
    return pos
    
def write_lammps_coord(file_dir,traj,nparticles,boxlength=None):
    traj=traj.reshape((-1,nparticles,3))
    with open(file_dir, 'a') as pos:
        for j in range(len(traj)):
                atom_index=np.arange(nparticles)+1
                type_index=np.ones(nparticles)
                config = np.column_stack((atom_index, type_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u','%u', '%.5f', '%.5f', '%.5f'])

def write_coord(file_dir,traj,nparticles,boxlength=None):
    traj=traj.reshape((-1,nparticles,3))
    with open(file_dir, 'a') as pos:
        for j in range(len(traj)):
                #U=LJ_potential(traj[j],boxlength,cutoff=2.7)
                pos.write('%d\n'%nparticles)
                pos.write(' Atoms\n')
                #pos.write('U: %d\n' % U)
                atom_index=np.ones(nparticles)
                config = np.column_stack((atom_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u', '%.5f', '%.5f', '%.5f'])
def read_input(dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dir)
    cfg.freeze()
    print(cfg)
    return cfg

def setup_lammps(input_dir,boxlength=None):
    lmp=lammps()
    lmp.file(input_dir)
    if boxlength is not None:
        boxhi=boxlength/2
        lmp.reset_box([-boxhi,-boxhi,-boxhi],[boxhi,boxhi,boxhi],0,0,0)
    return lmp

def force_matching(cfg,lmp,logp,x):
    predicted_force=torch.autograd.grad(logp,x,torch.ones_like(logp),retain_graph=True)[0]
    actual_force=[]
    with open(cfg.output.training_dir+"temp.xyz","w"):
      pass
    write_coord(cfg.output.training_dir+"temp.xyz",x.data.cpu(),cfg.dataset.nparticles)
    for i in range(len(x)):
        #lmp.create_atoms(cfg.dataset.nparticles, , x[i])
        lmp.command("read_dump %s %d x y z box no add yes format xyz"%(cfg.output.training_dir+"temp.xyz",i))
        lmp.command("run 0")
        actual_force.append(lmp.numpy.extract_fix("force", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)/cfg.dataset.kT)
    
    actual_force=torch.tensor(actual_force).reshape(-1,cfg.dataset.nparticles*3)
    #print("actual force:",actual_force[0])
    #print("predicted force:",predicted_force[0])
    return torch.mean(torch.linalg.norm((actual_force-predicted_force)/actual_force,dim=1)) 

def LJ_potential(particle_pos, boxlength, epsilon=1., sigma=1., cutoff=None):
    """Calculates Lennard_Jones force between particles
    Arguments:
        particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        boxlength: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
    Returns:
        total_force_on_particle: A tensor of shape (n_particles, n_dimensions)
        representing the total force on a particle
    """
    pair_dist = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
    to_subtract = ((torch.abs(pair_dist) > 0.5 * boxlength)
                   * torch.sign(pair_dist) * boxlength)
    pair_dist -= to_subtract
    distances = torch.linalg.norm(pair_dist.float(), axis=-1)
    scaled_distances = distances + (distances == 0)
    distances_inverse = 1/scaled_distances
    if cutoff is not None:
        distances_inverse = distances_inverse-(distances >cutoff)*distances_inverse
        pow_6 = torch.pow(sigma*distances_inverse, 6)
        pow_6_shift = (sigma/cutoff)**6
        pair_potential = epsilon * 4 * (torch.pow(pow_6, 2)
                                    - pow_6 - pow_6_shift**2+pow_6_shift)
    else:
        pair_potential = epsilon * 4 * (torch.pow(sigma/scaled_distances, 12)
                                    - torch.pow(sigma/scaled_distances, 6))
    pair_potential = pair_potential *distances_inverse*distances
    total_potential = torch.sum(pair_potential)/2 
    return total_potential    

def subsample(data,nsamples,device):
    total_n = list(data.size())[0]
    indices = torch.randint(total_n,[nsamples]).to(device)
    return data.index_select(0,indices)

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, input_left_bound=None, input_right_bound=None,
                      output_left_bound=1.,output_right_bound=1., inverse=False, min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    if input_left_bound is None:
        input_left_bound=output_left_bound
    if input_right_bound is None:
            input_right_bound=output_right_bound
    inside_intvl_mask = (inputs >= input_left_bound) & (inputs <= input_right_bound)
    outside_interval_mask = ~inside_intvl_mask
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
    #if any(inside_intvl_mask):
    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
            inputs=inputs[inside_intvl_mask],
            unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
            unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
            inverse=inverse,
            left=input_left_bound, right=input_right_bound, bottom=output_left_bound, top=output_right_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )
    return outputs, logabsdet

def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet
