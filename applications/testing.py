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
sys.path.append("/home/sherryli/xsli/MBAR")
from mbar.solve import solver
import pymbar
from bar import BAR
from config import get_cfg_defaults
from nf.flows import *
from nf.models import NormalizingFlowModel
from nf.base import EinsteinCrystal
import nf.utils as util
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import matplotlib.pyplot as plt


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
    N=cfg.dataset.nparticles*3
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)  
    if cfg.prior.type=="lattice":
        prior = EinsteinCrystal(cfg.prior.lattice_dir, alpha=cfg.prior.alpha,device=cfg.device)
    elif cfg.prior.type=="normal":
        prior = MultivariateNormal(torch.zeros(N).to(cfg.device), 0.5*torch.eye(N).to(cfg.device))
    if cfg.flow.type=="RealNVP":
        flows = [eval(cfg.flow.type)(dim=N,hidden_dim=cfg.flow.hidden_dim) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_AR":
        flows = [eval(cfg.flow.type)(dim=N, K=cfg.flow.nsplines, B=B,hidden_dim=cfg.flow.hidden_dim,device=cfg.device) for _ in range(cfg.flow.nlayers)]
    elif cfg.flow.type=="NSF_CL":
        x = [[0],[1],[2],[0,1],[1,2],[0,2]]
        mask= sum([x for _ in range(cfg.flow.nlayers//6+1)], [])[:cfg.flow.nlayers]
        flows = [eval(cfg.flow.type)(size=cfg.dataset.nparticles,dim=3, K=cfg.flow.nsplines, output_left_bound=-B,output_right_bound=B,hidden_dim=cfg.flow.hidden_dim, mask=mask[i],device=cfg.device) for i in range(cfg.flow.nlayers)]
    model = NormalizingFlowModel(prior, flows,cfg.device).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train_parameters.learning_rate)
    #scheduler = torch. optim.lr_scheduler.ExponentialLR(optimizer, cfg.train_parameters.lr_scheduler_gamma)
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

def generate_from_nf(cfg,model, prior, lmp, nsamples=50):
    #x, log_det ,z = model.sample(nsamples)
    z=prior.sample((nsamples,)).to(cfg.device)
    x, log_det = model.inverse(z)
    plt.hist(x[:,0].detach().cpu().numpy(),label="first layer")
    plt.hist(x[:,1].detach().cpu().numpy())
    plt.hist(x[:,2].detach().cpu().numpy())
    plt.legend()
    plt.savefig("dist.png")
    plt.show()
    log_px=prior.log_prob(z)-log_det
    pos_dir=cfg.output.testing_dir+ "generated_configs_%s.xyz"%cfg.dataset.name
    with open(pos_dir, 'w'):
        pass
    util.write_coord(pos_dir,x.data.cpu(),cfg.dataset.nparticles)
    energy=[]
    for i in range(len(x)):
        lmp.command("read_dump %s %d x y z box no add yes format xyz"%(pos_dir,i))
        lmp.command("run 1")
        energy.append(lmp.extract_variable("energy"))
    return x.data, log_px.data, -np.array(energy)/cfg.dataset.kT

def load_md_data(cfg,dir,model,prior,lmp,save_force=True):
    traj = MDA.coordinates.XYZ.XYZReader(dir)
    pos = torch.tensor(np.array([np.array(traj[i]) for i in range(len(traj))]),requires_grad=True).to(cfg.device)
    force_from_md=torch.from_numpy(np.load("md_data/force_100K_test.npy"))/cfg.dataset.kT
    z,_,log_det=model.forward(pos.reshape(len(traj),-1))
    q_nf=prior.log_prob(z)+log_det
    if save_force:
        force=torch.autograd.grad(q_nf,pos,torch.ones_like(q_nf))[0]
        lmp.command("read_dump md_data/fe_100K_test.xyz 0 x y z box no add yes format xyz")
        lmp.command("run 0")
        force_test=lmp.numpy.extract_fix("force", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)/cfg.dataset.kT
        print((force[0].cpu()-force_test)/force_test)
        #torch.save(cfg.output.testing_dir+"force_%s.pt"%cfg.dataset.name,torch.tensor(force[0]).data)
    energy=[]
    for i in range(len(traj)):
        lmp.command("read_dump %s %d x y z box no add yes format xyz"%(dir,i))
        lmp.command("run 1")
        energy.append(lmp.extract_variable("energy"))
    #print("check energy is correct:", pe-np.array(energy)/cfg.dataset.kT)
    return pos,q_nf.data,-np.array(energy)/cfg.dataset.kT

def plot_Q(cfg,Q):
    fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(12,6),tight_layout=True)
    ax1.plot(Q[0][:,0], Q[0][:,1],'.',color="darkgray")
    ax1.set_title("trajectory generated by NF")
    ax2.plot(Q[1][:,0], Q[1][:,1],'.',color="darkgray")
    ax2.set_title("trajectory from MD simulation")
    fig.supxlabel("logpx from NF")
    fig.supylabel("-potential (kT)")
    fig.savefig(cfg.output.testing_dir+"Q_%s.png"%cfg.dataset.name)
    plt.show()
    plt.close()

def metropolize(cfg,x,burnin=20):
    nsamples=x.size(dim=0)
    index=[False for i in range(nsamples)]
    frame=x[0].reshape(cfg.dataset.nparticles,3)
    energy=util.LJ_potential(frame, boxlength,cutoff=2.7)
    for i in range(nsamples):
        new_frame=x[i].reshape(cfg.dataset.nparticles,3)
        new_energy=util.LJ_potential(new_frame, boxlength,cutoff=2.7)
        acc_prob=torch.exp(energy-new_energy)
        if torch.rand(1)<acc_prob:
            frame=new_frame
            energy=new_energy
            if i>burnin:
                index[i]=True
    return index

if __name__ == "__main__":
    name=sys.argv[1]
    cfg=read_input("input/%s.yaml"%name)
    model,optimizer,scheduler,training_data,logger,boxlength = setup_model(cfg,training=False)
    lmp=setup_lammps("md_data/in.lmp",boxlength)
    nf = torch.load("trained_models/%s.pth"%name,map_location='cpu')
    #nf = torch.load("saved_models/%s3.pth"%name,map_location='cpu')
    np.savetxt(cfg.output.testing_dir+"loss_%s.dat"%cfg.dataset.name,torch.Tensor(nf["loss"]).cpu().numpy())
    model.load_state_dict(nf["model"],strict=False)
    model=model.to(cfg.device)
    #sample_prior = EinsteinCrystal(cfg.prior.lattice_dir, alpha=200,device=cfg.device)
    sample_prior=model.prior
    traj0,q00,q01=generate_from_nf(cfg,model,sample_prior,lmp,nsamples=1000)
    q00=q00.cpu().numpy()
    traj0=traj0.cpu().reshape(-1,cfg.dataset.nparticles,3)
    Q=[]
    Q.append(np.transpose(np.vstack((q00,q01))))
    with open(cfg.output.testing_dir+"Q0_%s.dat"%cfg.dataset.name, "w") as f:
        np.savetxt(f, Q[0])
    traj1,q10,q11=load_md_data(cfg,cfg.dataset.testing_dir,model,sample_prior,lmp)
    q10=q10.cpu().numpy()
    Q.append(np.transpose(np.vstack((q10,q11))))
    with open(cfg.output.testing_dir+"Q1_%s.dat"%cfg.dataset.name,"w"):
        np.savetxt(cfg.output.testing_dir+"Q1_%s.dat"%cfg.dataset.name,Q[1])
    Nk=np.array([len(Q[0]),len(Q[1])])
    u=np.vstack((-Q[0],-Q[1])).transpose()
    mbar=pymbar.mbar.MBAR(u,Nk)
    free_energy=mbar.getFreeEnergyDifferences(return_dict=True)
    plot_Q(cfg,Q)
    print(free_energy)
    print("Absolute free energy per particle (eV):",free_energy['Delta_f'][0,1]/cfg.dataset.nparticles*cfg.dataset.kT)
    to_subtract_0=np.min(Q[0][:,0])
    to_subtract_1=np.min(Q[0][:,1])
    Q[0][:,0]-=to_subtract_0
    Q[0][:,1]-=to_subtract_1
    Q[1][:,1]-=to_subtract_1
    Q[1][:,0]-=to_subtract_0
    c = solver([np.exp(Q[0]),np.exp(Q[1])],niter=10).norm_const()
    print((to_subtract_0-to_subtract_1+np.log(c[0]))/cfg.dataset.nparticles*cfg.dataset.kT)
    print("bar estimate:",(to_subtract_0-to_subtract_1+BAR(Q[0][:,0]-Q[0][:,1],-Q[1][:,0]+Q[1][:,1]))/cfg.dataset.nparticles*cfg.dataset.kT)
    print("simple estimate from md data:", (to_subtract_0-to_subtract_1)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[1][:,1]-Q[1][:,0])))/cfg.dataset.nparticles*cfg.dataset.kT)
    print("simple estimate from generated data:", (to_subtract_1-to_subtract_0)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[0][:,0]-Q[0][:,1])))/cfg.dataset.nparticles*cfg.dataset.kT)

    '''
    nbatches=4
    batchsize=1000
    pos=[]
    logp=[]
    pot=[]
    for i in range(nbatches):
        z = sample_prior.sample((batchsize,))
        x, log_det = model.inverse(z)
        index =metropolize(cfg,x.data)
        pos.append(x.data[index])
        logp.append(sample_prior.log_prob(z[index])-log_det[index])
        pot.append(torch.Tensor([util.LJ_potential(x[i].reshape(-1,3), boxlength,cutoff=2.7) for i in np.arange(batchsize)[index]])/cfg.dataset.kT)
    '''