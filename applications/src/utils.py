
import matplotlib.pyplot as plt
import torch
import MDAnalysis as MDA
import numpy as np

def plot_Q(cfg,Q,split=False,save=True):
    if split:
        fig,(ax1,ax2)=plt.subplots(1,2,sharex=True, sharey=True,figsize=(12,6),tight_layout=True)
        ax1.plot(Q[0][:,0], Q[0][:,1],'.',color="darkgray")
        ax1.set_title("trajectory generated by NF")
        ax2.plot(Q[1][:,0], Q[1][:,1],'.',color="darkgray")
        ax2.set_title("trajectory from MD simulation")
        fig.supxlabel("logpx from NF")
        fig.supylabel("-potential (kT)")
        if save:
            fig.savefig(cfg.output.testing_dir+"Q_%s.png"%cfg.dataset.name)
        plt.show()
        plt.close()
    else:
        plt.plot(Q[0][:,0], Q[0][:,1],'.',color="darkblue",label="NF traj")
        plt.plot(Q[1][:,0], Q[1][:,1],'.',color="darkgray",label="MD traj")
        plt.xlabel("logpx from NF")
        plt.ylabel("-potential (kT)")
        plt.legend()
        if save:
            plt.savefig(cfg.output.testing_dir+"Q_%s.png"%cfg.dataset.name)
        plt.show()
        plt.close()

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
                atom_index=np.arange(nparticles)
                type_index=np.ones(nparticles)
                config = np.column_stack((atom_index, type_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u','%u', '%.5f', '%.5f', '%.5f'])

def write_coord(file_dir,traj,nparticles,boxlength=None,append=False):
    traj=traj.reshape((-1,nparticles,3))
    if not append:
        with open(file_dir, 'w') as pos:
            pass
    with open(file_dir, 'a') as pos:
        for j in range(len(traj)):
                #U=LJ_potential(traj[j],boxlength,cutoff=2.7)
                pos.write('%d\n'%nparticles)
                pos.write(' Atoms\n')
                #pos.write('U: %d\n' % U)
                atom_index=np.ones(nparticles)
                config = np.column_stack((atom_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u', '%.5f', '%.5f', '%.5f'])

def metropolize(cfg,potential,x,burnin=20):
    x=x.cpu().detach().reshape(-1,cfg.dataset.nparticles,3)
    nsamples=x.size(dim=0)
    index=[False for i in range(nsamples)]
    frame=x[0]
    energy=potential.potential(frame)/cfg.dataset.kT
    energy_list=[]
    for i in range(nsamples):
        new_frame=x[i]
        new_energy=potential.potential(new_frame)/cfg.dataset.kT
        acc_prob=torch.exp(energy-new_energy)
        if torch.rand(1)<acc_prob:
            frame=new_frame
            energy=new_energy
            if i>burnin:
                index[i]=True
            energy_list.append(energy)
    return x[index],energy_list

def subsample(data,nsamples,device="cpu",random=True):
    if random:
        total_n = len(data)
        indices = torch.randint(total_n,[nsamples]).to(device)
        return data.index_select(0,indices)
    else:
        return data[:nsamples]