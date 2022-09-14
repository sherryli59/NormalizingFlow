import torch
import setup
import sys
sys.path.append("../../")
from bar import BAR
from relaxation import * 
from utils import *
sys.path.append("/home/groups/rotskoff/sherry/MBAR")
from mbar.solve import MBAR
import numpy as np



def generate_from_nf(cfg, model, prior, nsamples=50):
    #x, log_px , _= model.sample(nsamples)
    z=prior.sample((nsamples,))
    x, log_det = model.inverse(z)
    print("prior logprob of generated data",torch.mean(prior.log_prob(z)))
    log_px=prior.log_prob(z)-log_det
    return x, log_px
    
def compute_fe_diff(cfg,model,potential, nsamples, relaxation=False,return_err=False):
    traj0,q00=generate_from_nf(cfg, model,model.prior, nsamples)
    if relaxation: 
        traj0, q00, q01 = relaxation_step(cfg,model,potential,traj0)
    else:
        q01=-torch.tensor(potential.potential(traj0.reshape(-1,cfg.dataset.nparticles,cfg.dataset.dim))/cfg.dataset.kT).to(cfg.device)
    Q0=torch.transpose(torch.vstack((q00,q01)),0,1)
    traj1=potential.sample(nsamples) 
    if relaxation:
        traj1, q10,q11 = relaxation_step(cfg,model,potential,traj1)
        #util.write_coord(cfg.output.testing_dir+"relaxed_configs_%s.xyz"%cfg.dataset.name,traj1.detach(),cfg.dataset.nparticles)
    else:
        q10 = torch.zeros(nsamples)
        z,_,log_det=model.forward(traj1.reshape(len(traj1),-1))
        q10 += model.prior.log_prob(z)+log_det
        q11=-torch.tensor(potential.potential(traj1.reshape(-1,cfg.dataset.nparticles,cfg.dataset.dim))/cfg.dataset.kT).to(cfg.device)
    Q1= torch.transpose(torch.vstack((q10,q11)),0,1)
    Q=torch.stack((Q0,Q1))
    Q_np=Q.detach().cpu().numpy()
    np.save(cfg.output.testing_dir+"Q_%s.dat"%cfg.dataset.name, Q_np)
    plot_Q(cfg,Q_np,split=False)
    #print("prior logprob of testing data:",torch.mean(model.prior.log_prob(z)))
    to_subtract_0=torch.min(Q[1][:,0])
    to_subtract_1=torch.min(Q[1][:,1])
    Q[0][:,0]-=to_subtract_0
    Q[0][:,1]-=to_subtract_1
    Q[1][:,1]-=to_subtract_1
    Q[1][:,0]-=to_subtract_0
    solver = MBAR(-Q.cpu())
    c=solver.norm_const(niter=40)
    emus=(to_subtract_0-to_subtract_1+torch.log(c[0].detach())-torch.log(c[1].detach()))/cfg.dataset.nparticles*cfg.dataset.kT
    Q=Q.detach().cpu().numpy()
    plot_Q(cfg,Q,save=False)
    bar=(to_subtract_0-to_subtract_1+BAR(Q[0][:,0]-Q[0][:,1],-Q[1][:,0]+Q[1][:,1]))/cfg.dataset.nparticles*cfg.dataset.kT
    md = (to_subtract_0-to_subtract_1)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[1][:,1]-Q[1][:,0])))/cfg.dataset.nparticles*cfg.dataset.kT
    nf = -((to_subtract_1-to_subtract_0)/cfg.dataset.nparticles*cfg.dataset.kT-np.log(np.mean(np.exp(Q[0][:,0]-Q[0][:,1])))/cfg.dataset.nparticles*cfg.dataset.kT)
    if return_err:
        c_err, c_err_contribs, _ = solver.norm_const_err(c)
        return  bar, c, torch.sqrt(c_err), c_err_contribs
    return bar, md, nf, emus


def main():
    name=sys.argv[1]
    cfg=setup.read_input("input/%s.yaml"%name)
    model_dir="trained_models"
    model,potential = setup.load_model(name,cfg,model_dir)
    bar, md, nf, c = compute_fe_diff(cfg,model,potential, nsamples=500,relaxation=True, return_err=False)
    print(bar,md,nf,c)
    
if __name__ == "__main__":
    main()