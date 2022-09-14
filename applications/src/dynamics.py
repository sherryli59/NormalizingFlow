import torch

def relaxation_step(cfg,model,simulation,traj):
    input_dir=cfg.dataset.input_dir
    LMP=LAMMPS(input_dir,cell_len= cfg.dataset.cell_len)
    q_fe = []
    q_fe_before = []
    q_learned = []
    logp_v = []
    traj_new = []
    for i in range(len(traj)):
        init_pos=traj[i].cpu().detach().numpy()
        LMP.set_position(init_pos)
        q_fe_before.append(-LMP.get_potential()/cfg.dataset.kT)
        run=HMC(LMP, beta=1/cfg.dataset.kT, mass=torch.tensor([55.845]),path_len=12, init_beta=1/cfg.dataset.kT/1000) 
        position,potential,log_prob = run.run_sim()
        q_fe.append(-potential/cfg.dataset.kT)        
        traj_new.append(torch.tensor(position.flatten()))
        q_learned.append(integrate_out_v(cfg,model,simulation,run,position))
        logp_v.append(torch.tensor(log_prob))
    print("average potential before relaxation",torch.mean(torch.tensor(q_fe_before)))    
    print("average potential after relaxation",torch.mean(torch.tensor(q_fe)))
    return torch.stack(traj_new).float().to(cfg.device), torch.tensor(q_learned).to(cfg.device), torch.tensor(q_fe).to(cfg.device)


def integrate_out_v(cfg,model,simulation,hmc,frame,npoints=10):
    v_list=hmc.v_dist.sample((npoints,))
    log_pv_list=[]
    traj=[]
    for v in v_list:
        simulation.set_position(frame)
        position,_,log_pv = hmc.run_sim(v)
        #log_pv_list.append(log_pv)
        traj.append(torch.tensor(position.flatten()).float())
    q_nf=model.evaluate(torch.stack(traj).to(cfg.device))
    return torch.logsumexp(q_nf,0)-torch.log(torch.tensor(npoints))
'''
def relaxation_step(cfg,model,simulation,traj):
    q_fe_before = []
    q_fe = []
    q_learned = []
    logp_v = []
    traj_new = []
    for i in range(len(traj)):
        init_pos=traj[i]
        simulation.set_position(init_pos)
        q_fe_before.append(-simulation.get_potential()/cfg.dataset.kT)
        run=HMC(simulation,beta=1, path_len=1,dim=cfg.dataset.dim)
        position,potential,log_prob = run.run_sim()
        q_fe.append(-potential/cfg.dataset.kT)
        traj_new.append(torch.tensor(position.flatten()))
        q_learned.append(integrate_out_v(cfg,model,simulation,run,position))
        logp_v.append(torch.tensor(log_prob))
    print("average potential before relaxation",torch.mean(torch.tensor(q_fe_before)))    
    print("average potential after relaxation",torch.mean(torch.tensor(q_fe)))
    return torch.stack(traj_new).float().to(cfg.device), torch.tensor(q_learned).to(cfg.device), torch.tensor(q_fe).to(cfg.device)
'''

def collect_hmc_data(cfg,model,hmc, burnin=100):
    samples,_,z = model.sample(1)
    util.write_coord(cfg.output.training_dir+"generated_configs.xyz",samples,cfg.dataset.nparticles)
    traj, _, _, acc_prob = hmc.hmc(epochs=500, init_pos = samples.cpu())
    print("acceptance prob:", acc_prob)
    util.write_coord(cfg.output.training_dir+"relaxed_configs.xyz",traj,cfg.dataset.nparticles)
    return traj[burnin:], acc_prob
