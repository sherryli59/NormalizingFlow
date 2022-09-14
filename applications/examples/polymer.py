import torch
import sys
sys.path.append("../../")   
sys.path.append("../")
from src import setup   
import numpy as np 

def generate_from_nf(cfg, model, nsamples=50):
    x, log_px , _= model.sample(nsamples)
    save_field(cfg,x.detach().numpy()) 
    return x, log_px

def plot_field(x):
    x= x.reshape(2,32,32) 
    omega_plus = x[0]  
    omega_minus= x[1]   
    plt.imshow(omega_plus)
    plt.savefig("omega_plus.png")
    plt.close()
    plt.imshow(omega_minus)
    plt.savefig("omega_minus.png")
    plt.close()

def save_field(cfg,x):
    x=x.reshape(-1,2,32,32)
    np.save(cfg.output.testing_dir+"generated_fields.npy",x)

def main(mode):
    name="Polymer"
    cfg=setup.read_input("../input/%s.yaml"%name)
    if mode == "training":
        model,optimizer,scheduler,logger,potential = setup.setup_model(cfg)
        train(cfg,model,optimizer,scheduler,logger,potential)
    if mode == "testing":
        model,potential = setup.load_model(name,cfg,cfg.output.best_model_dir)
        nsamples = 100
        x1, q1 =generate_from_nf(cfg,model,nsamples=nsamples)
        x2 = potential.sample(nsamples)
        x2 = x2.reshape(len(x2),-1)
        q2 = model.evaluate(x2)
        print("logp of generated data vs testing data:", torch.mean(q1),torch.mean(q2))

if __name__ == "__main__":
    main(mode="testing")