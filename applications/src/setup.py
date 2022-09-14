import numpy as np
import os
import logging
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
import sys
sys.path.append("../../")
from src.config import get_cfg_defaults
from src import systems
from nf.flows import *
from nf.models import NormalizingFlowModel



def parse_potential(name,config, device):
    if name =="GaussianMixture":
        potential=systems.GaussianMixture(config.centers,config.vars,config.nparticles,config.dim,device=device)
    elif name =="LJ":
        potential=systems.LJ(pos_dir=config.data, boxlength=config.boxlength, device=device, sigma=config.sigma, epsilon=config.epsilon, cutoff=config.cutoff, shift=config.shift)
    elif name =="EinsteinCrystal":
        potential=systems.EinsteinCrystal(config.centers,config.dim,boxlength=config.boxlength, alpha=config.alpha,device=device)
        
    elif name =="Normal":
        N=config.nparticles* config.dim
        if config.vars == None:
            vars = 1
        else: vars = config.vars
        potential = MultivariateNormal(torch.zeros(N).to(device), vars*torch.eye(N).to(device))
    elif name == "Fe":
        potential = systems.Fe(config.input_dir, pos_dir=config.data, cell_len=config.cell_len, device=device)
    
    elif name == "SimData":
        potential = systems.SimData(pos_dir=config.data, device=device,data_type=config.type)
    return potential

def setup_model(cfg, mode="training"):
    if cfg.dataset.rho is not None:
        B=(cfg.dataset.nparticles/(8*cfg.dataset.rho))**(1/3)
    elif cfg.dataset.ncellx is not None:
        B=cfg.dataset.ncellx*cfg.dataset.cell_len/2

    if cfg.dataset.boxlength is None:
        cfg.dataset.boxlength = 2*B


    N=cfg.dataset.nparticles* cfg.dataset.dim
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True

    prior = parse_potential(name=cfg.prior.type, config=cfg.prior, device=cfg.device)

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
    if cfg.train_parameters.scheduler == "exponential":
            scheduler = torch. optim.lr_scheduler.ExponentialLR(optimizer, cfg.train_parameters.lr_scheduler_gamma)
    elif cfg.train_parameters.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train_parameters.max_epochs)

    if not(os.path.exists(cfg.output.model_dir)):
        os.mkdir(cfg.output.model_dir)
    if not(os.path.exists(cfg.output.training_dir)):
        os.mkdir(cfg.output.training_dir)
    if not(os.path.exists(cfg.output.testing_dir)):
        os.mkdir(cfg.output.testing_dir)
    if not(os.path.exists(cfg.output.model_dir)):
        os.mkdir(cfg.output.model_dir)

    if mode=="training":
            cfg.dataset.data= cfg.dataset.training_data
    elif mode=="testing":
        cfg.dataset.data= cfg.dataset.testing_data
    potential = parse_potential(name=cfg.dataset.potential, config=cfg.dataset, device=cfg.device)

    return model,optimizer,scheduler,logger,potential

def read_input(dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dir)
    print(cfg)
    return cfg

def reverseKL(model,potential,nsamples):
    z = model.prior.sample((nsamples,))
    x, log_det = model.inverse(z)
    log_prob = model.prior.log_prob(z)-log_det
    return -torch.mean(potential.log_prob(x))+torch.mean(log_prob)

def KL(model,potential,nsamples):
    x = potential.sample(nsamples,flatten=True)
    z, prior_logprob, log_det = model(x)
    logprob = prior_logprob + log_det
    return -torch.mean(logprob)+torch.mean(potential.log_prob(x))

def load_model(name,cfg,model_dir):
    nf = torch.load("%s/%s.pth"%(model_dir,name),map_location='cpu')
    np.savetxt(cfg.output.testing_dir+"loss_%s.dat"%cfg.dataset.name,torch.Tensor(nf["loss"]).cpu().numpy())

    model,optimizer,scheduler,logger,potential = setup_model(cfg,mode="testing")
    model.load_state_dict(nf["model"],strict=False)
    model=model.to(cfg.device) 
    return model,potential
    