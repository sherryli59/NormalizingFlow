import numpy as np
import os
import itertools
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
import pymbar
from config import get_cfg_defaults
from nf.flows import *
from nf.models import NormalizingFlowModel
from nf.base import EinsteinCrystal, GaussianMixture
import nf.utils as util
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY

def read_input(dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dir)
    cfg.freeze()
    print(cfg)
    return cfg

def setup_model(cfg,training=True):
    if cfg.dataset.rho is not None:
        B=(cfg.dataset.nparticles/(8*cfg.dataset.rho))**(1/3)
    else:
        B=cfg.dataset.ncellx*cfg.dataset.cell_len/2
    boxlength=2*B
    N=cfg.dataset.nparticles*cfg.dataset.dim
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)  
    if cfg.prior.type=="lattice":
        prior = EinsteinCrystal(cfg.prior.lattice_dir, alpha=cfg.prior.alpha,device=cfg.device)
    elif cfg.prior.type=="normal":
        prior = MultivariateNormal(torch.zeros(N).to(cfg.device), 0.5*torch.eye(N).to(cfg.device))
    elif cfg.prior.type=="gaussian_mix":
        prior = GaussianMixture(cfg.prior.centers, cfg.prior.vars, cfg.dataset.nparticles, cfg.dataset.dim)
    if cfg.flow.type=="RealNVP":
        flows = [eval(cfg.flow.type)(dim=N,hidden_dim=cfg.flow.hidden_dim) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_AR":
        flows = [eval(cfg.flow.type)(dim=N, K=cfg.flow.nsplines, B=B,hidden_dim=cfg.flow.hidden_dim,periodic=cfg.dataset.periodic,device=cfg.device) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_CL":
        x = [[0],[1],[2],[0,1],[1,2],[0,2]]
        mask= sum([x for _ in range(cfg.flow.nlayers//6+1)], [])[:cfg.flow.nlayers]
        flows = [eval(cfg.flow.type)(size=cfg.dataset.nparticles,dim=3, K=cfg.flow.nsplines, output_left_bound=-B,output_right_bound=B,hidden_dim=cfg.flow.hidden_dim, mask=mask[i],device=cfg.device) for i in range(cfg.flow.nlayers)]
    model = NormalizingFlowModel(prior, flows,cfg.device).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_parameters.learning_rate)
    if cfg.train_parameters.scheduler == "exponential":
        scheduler = torch. optim.lr_scheduler.ExponentialLR(optimizer, cfg.train_parameters.lr_scheduler_gamma)
    elif cfg.train_parameters.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train_parameters.max_epochs)
    if training:
        training_data = util.load_position(cfg.dataset.training_dir).to(cfg.device)
    else:
        training_data=None
    if not(os.path.exists(cfg.output.model_dir)):
        os.mkdir(cfg.output.model_dir)
    return model,optimizer,scheduler,training_data,logger,boxlength


def setup_lammps(input_dir,boxlength=None):
    lmp=lammps()
    lmp.file(input_dir)
    if boxlength is not None:
        boxhi=boxlength/2
        lmp.reset_box([-boxhi,-boxhi,-boxhi],[boxhi,boxhi,boxhi],0,0,0)
    return lmp

def force_matching(cfg,lmp,logp,x):
    predicted_force=torch.autograd.grad(logp,x,torch.ones_like(logp),create_graph=True)[0]
    actual_force=[]
    with open(cfg.output.training_dir+"temp.xyz","w"):
      pass
    util.write_coord(cfg.output.training_dir+"temp.xyz",x.data.cpu(),cfg.dataset.nparticles)
    for i in range(len(x)):
        #lmp.create_atoms(cfg.dataset.nparticles, , x[i])
        lmp.command("read_dump %s %d x y z box no add yes format xyz"%(cfg.output.training_dir+"temp.xyz",i))
        lmp.command("run 0")
        actual_force.append(lmp.numpy.extract_fix("force", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)/cfg.dataset.kT)
    
    actual_force=torch.tensor(actual_force).reshape(-1,cfg.dataset.nparticles*3)
    #print("actual force:",actual_force[0])
    #print("predicted force:",predicted_force[0])
    return torch.mean(torch.linalg.norm(actual_force-predicted_force,dim=1)) 

def train(cfg,model,optimizer,scheduler,training_data,logger,lmp):
    losses=[]
    max_logprob=140
    lamb=0.5
    
    for i in range(cfg.train_parameters.max_epochs):
        optimizer.zero_grad()
        x = util.subsample(training_data,nsamples=cfg.train_parameters.batch_size,device=cfg.device)
        x.requires_grad_()
        z, prior_logprob, log_det = model(x)
        #x1, log_det1 =model.inverse(z)
        #print("check the map is invertible", torch.linalg.norm(x1-x), torch.linalg.norm(log_det1+log_det))
        logprob = prior_logprob + log_det
        forward_loss=-torch.mean(logprob)
        #fm=force_matching(cfg,lmp,logprob,x)
        #print(fm)
        loss = forward_loss
        #loss=fm
        losses.append(loss.mean().data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % cfg.train_parameters.output_freq == 0:
            logger.info(f"Iter: {i}\t" +
                        f"Loss: {loss.mean().data:.2f}\t" +
                        f"Logprob: {logprob.mean().data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}")
            samples,_,z = model.sample(1)
            util.write_coord(cfg.output.training_dir+"generated_configs.xyz",samples,cfg.dataset.nparticles)
            if (i>100) and (-forward_loss>max_logprob):
                max_logprob=-forward_loss
                torch.save({"model":model.state_dict(),"optim": optimizer.state_dict(),
                            "loss":losses},cfg.output.model_dir+cfg.dataset.name+'%d.pth'% (i//cfg.train_parameters.output_freq))
        

if __name__ == "__main__":
    name=sys.argv[1]
    cfg=read_input("input/%s.yaml"%name)
    model,optimizer,scheduler,training_data,logger,boxlength = setup_model(cfg)
    lmp=setup_lammps("md_data/in.lmp",boxlength)
    #nf = torch.load("trained_models/%s.pth"%name,map_location='cpu')
    #nf = torch.load("saved_models/%s78.pth"%name,map_location='cpu')
    #model.load_state_dict(nf["model"],strict=False)
    model=model.to(cfg.device)
    train(cfg,model,optimizer,scheduler,training_data,logger,lmp)