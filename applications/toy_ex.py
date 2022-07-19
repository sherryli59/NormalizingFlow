import numpy as np
import os
import itertools
import logging
import matplotlib.pyplot as plt
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
from nf.base import EinsteinCrystal
import nf.utils as util
import random
import systems
from bar import BAR
sys.path.append("/home/sherryli/xsli/MBAR")
from mbar.solve import solver

def setup_model(cfg):
    if cfg.dataset.rho is not None:
        B=(cfg.dataset.nparticles/(8*cfg.dataset.rho))**(1/3)
    else:
        B=cfg.dataset.ncellx*cfg.dataset.cell_len/2
    boxlength=2*B
    N=cfg.dataset.nparticles* cfg.dataset.dim
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)  
    if cfg.prior.type=="lattice":
        prior = EinsteinCrystal(cfg.prior.lattice_dir, alpha=cfg.prior.alpha,device=cfg.device)
    elif cfg.prior.type=="normal":
        prior = MultivariateNormal(torch.zeros(N).to(cfg.device), 0.1*B*torch.eye(N).to(cfg.device))
    elif cfg.prior.type=="gaussian_mix":
        prior = systems.GaussianMixture(cfg.prior.centers,cfg.prior.vars,cfg.dataset.nparticles,cfg.dataset.dim)
    if cfg.flow.type=="RealNVP":
        flows = [eval(cfg.flow.type)(dim=N,hidden_dim=cfg.flow.hidden_dim) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_AR":
        flows = [eval(cfg.flow.type)(dim=N, K=cfg.flow.nsplines, B=B,hidden_dim=cfg.flow.hidden_dim,device=cfg.device) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_CL":
        x = [[0],[1],[2],[0,1],[1,2],[0,2]]
        mask= sum([x for _ in range(cfg.flow.nlayers//6+1)], [])[:cfg.flow.nlayers]
        flows = [eval(cfg.flow.type)(size=cfg.dataset.nparticles,dim=3, K=cfg.flow.nsplines, B=B,hidden_dim=cfg.flow.hidden_dim, mask=mask[i],device=cfg.device) for i in range(cfg.flow.nlayers)]
    model = NormalizingFlowModel(prior, flows,cfg.device).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_parameters.learning_rate)
    scheduler = torch. optim.lr_scheduler.ExponentialLR(optimizer, cfg.train_parameters.lr_scheduler_gamma)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train_parameters.max_epochs)
    if cfg.dataset.training_dir is not None:
        training_data = util.load_position(cfg.dataset.training_dir).to(cfg.device)
    else:
        training_data=None
    if not(os.path.exists(cfg.output.model_dir)):
        os.mkdir(cfg.output.model_dir)
    if not(os.path.exists(cfg.output.training_dir)):
        os.mkdir(cfg.output.training_dir)
    if not(os.path.exists(cfg.output.testing_dir)):
        os.mkdir(cfg.output.testing_dir)
    if cfg.dataset.potential=="GaussianMixture":
        potential=systems.GaussianMixture(cfg.dataset.centers,cfg.dataset.vars,cfg.dataset.nparticles,torch.tensor(cfg.dataset.centers).shape[1])
        
    return model,optimizer,scheduler,training_data,logger,boxlength,potential

def read_input(dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dir)
    cfg.freeze()
    print(cfg)
    return cfg

def generate_from_nf(model, prior, nsamples=50):
    x, log_det ,z = model.sample(nsamples)
    z=prior.sample((nsamples,))
    x, log_det = model.inverse(z)
    log_px=prior.log_prob(z)-log_det
    return x.data, log_px.data

def compute_fe_diff(cfg,model,nsamples):
    nf = torch.load("trained_models_new/%s.pth"%name,map_location='cpu')
    #nf = torch.load("saved_models/%s14.pth"%name,map_location='cpu')
    np.savetxt(cfg.output.testing_dir+"loss_%s.dat"%cfg.dataset.name,torch.Tensor(nf["loss"]).cpu().numpy())
    model.load_state_dict(nf["model"],strict=False)
    model=model.to(cfg.device)
    traj0,q00=generate_from_nf(model,model.prior, nsamples)
    #plot(traj0)
    q00=q00.cpu().numpy()
    #traj0=traj0.cpu().reshape(-1,cfg.dataset.nparticles,cfg.dataset.dim)
    q01=potential.log_prob(traj0).detach().cpu().numpy()
    Q=[]
    Q.append(np.transpose(np.vstack((q00,q01))))
    traj1=potential.sample(nsamples)
    #plot(traj1)
    z,_,log_det=model.forward(traj1.reshape(len(traj1),-1))
    log_det=0
    q10=model.prior.log_prob(z)-log_det
    q10=q10.detach().cpu().numpy()
    q11=potential.log_prob(traj1).detach().cpu().numpy()
    q10=q11
    Q.append(np.transpose(np.vstack((q10,q11))))
    with open(cfg.output.testing_dir+"Q0_%s.dat"%cfg.dataset.name, "w") as f:
            np.savetxt(f, Q[0])
    with open(cfg.output.testing_dir+"Q1_%s.dat"%cfg.dataset.name,"w"):
            np.savetxt(cfg.output.testing_dir+"Q1_%s.dat"%cfg.dataset.name,Q[1])
    to_subtract_0=np.min(Q[0][:,0])
    to_subtract_1=np.min(Q[0][:,1])
    Q[0][:,0]-=to_subtract_0
    Q[0][:,1]-=to_subtract_1
    Q[1][:,1]-=to_subtract_1
    Q[1][:,0]-=to_subtract_0
    bar=(to_subtract_0-to_subtract_1+BAR(Q[0][:,0]-Q[0][:,1],-Q[1][:,0]+Q[1][:,1]))/cfg.dataset.nparticles*cfg.dataset.kT
    md = (to_subtract_0-to_subtract_1)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[1][:,1]-Q[1][:,0])))/cfg.dataset.nparticles*cfg.dataset.kT
    nf = -((to_subtract_1-to_subtract_0)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[0][:,0]-Q[0][:,1])))/cfg.dataset.nparticles*cfg.dataset.kT)
    c = solver([np.exp(Q[0]),np.exp(Q[1])],niter=40).norm_const()
    emus=(to_subtract_0-to_subtract_1+np.log(c[0]))/cfg.dataset.nparticles*cfg.dataset.kT
    return bar, md, nf, emus

def plot_trend(cfg,x,y,err,xlabel=None,title=None,log_base=None):
    fig, ax=plt.subplots(1,1,tight_layout=True)
    ax.scatter(x,y,marker='.')
    ax.errorbar(x, y,yerr=err,fmt=".")
    if log_base is not None:
            ax.set_xscale('log',base=log_base)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("FE difference")
    if title is not None:
        ax.set_title(title)
    ax.plot(x,np.zeros(len(x)),label="y=0")
    ax.legend()
    fig.savefig("testing/gaussian_rnvp/"+title+".png")
    print(cfg.output.testing_dir+title+".png")
    fig.savefig(cfg.output.testing_dir+title+".png")
    plt.show()
    plt.close() 

def size_dependence(cfg,model,nsamples,ntrials):
    fe_bar_list=[]
    fe_bar_err_list=[]
    fe_md_list=[]
    fe_md_err_list=[]
    fe_nf_list=[]
    fe_nf_err_list=[]
    fe_emus_list=[]
    fe_emus_err_list=[]
    for i in nsamples:
        fe_bar=[]
        fe_md=[]
        fe_nf=[]
        fe_emus=[]
        for _ in range(ntrials):
            bar,md,nf,emus = compute_fe_diff(cfg,model,i)
            fe_bar.append(bar)
            fe_md.append(md)
            fe_nf.append(nf)
            fe_emus.append(emus)
        fe_bar_list.append(np.mean(np.array(fe_bar)))
        fe_bar_err_list.append(np.std(np.array(fe_bar)))
        fe_md_list.append(np.mean(np.array(fe_md)))
        fe_md_err_list.append(np.std(np.array(fe_md)))
        fe_nf_list.append(np.mean(np.array(fe_nf)))
        fe_nf_err_list.append(np.std(np.array(fe_nf)))
        fe_emus_list.append(np.mean(np.array(fe_emus)))
        fe_emus_err_list.append(np.std(np.array(fe_emus)))
    return np.array(fe_bar_list),np.array(fe_bar_err_list), np.array(fe_md_list),np.array(fe_md_err_list), np.array(fe_nf_list),np.array(fe_nf_err_list),np.array(fe_emus_list),np.array(fe_emus_err_list)

if __name__ == "__main__":
    name=sys.argv[1]
    cfg=read_input("input/%s.yaml"%name)
    model,optimizer,scheduler,training_data,logger,boxlength,potential = setup_model(cfg)
    nsamples=2**(np.arange(0,8,2)+6)
    ntrials=20
    bar, bar_err, md, md_err, nf, nf_err, emus, emus_err=size_dependence(cfg,model,nsamples,ntrials)
    plot_trend(cfg, nsamples,np.array(bar),np.array(bar_err),"number of samples","bar_estimate",log_base=2)
    plot_trend(cfg, nsamples,np.array(md),np.array(md_err),"number of samples","simple_estimate_from_md_data",log_base=2)
    plot_trend(cfg, nsamples,np.array(nf),np.array(nf_err),"number of samples","simple_estimate_from_generated_data",log_base=2)
    plot_trend(cfg, nsamples,np.array(emus),np.array(emus_err),"number of samples","emus_estimate",log_base=2)