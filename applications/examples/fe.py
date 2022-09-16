import torch
import sys
sys.path.append("../../")   
sys.path.append("../")
from src import setup   
import numpy as np 
from src.train import train
from src.test import fe_diff

def generate_from_nf(model, nsamples=50,batchsize=50):
    for i in range(nsamples // batchsize):
        x, log_px , _= model.sample(batchsize)
        if i == 0:
            x_list = x
            log_px_list = log_px
        else:
            x_list = torch.cat((x_list,x),axis=0)
            log_px_list = torch.cat((log_px_list,log_px),axis=0)
    return x_list, log_px_list

def evaluate(model,x, batchsize=50):
    for i in range(len(x)// batchsize):
        xi = x[i*batchsize:(i+1)*batchsize]
        if i == 0:
            log_px_list = model.evaluate(xi)
        else:
            log_px_list = torch.cat((log_px_list,model.evaluate(xi)),axis=0)
    return log_px_list

def main(mode):
    name=sys.argv[1]
    cfg=setup.read_input("../input/%s.yaml"%name)
    if mode == "training":
        model,optimizer,scheduler,logger,potential = setup.setup_model(cfg)
        train(cfg,model,optimizer,scheduler,logger,potential)
    if mode == "testing":
        model,potential = setup.load_model(name,cfg,cfg.output.best_model_dir)
        nsamples = 2000
        x1, q1 =generate_from_nf(model,nsamples=nsamples,batchsize=500)
        x2 = potential.sample(nsamples)
        x2 = x2.reshape(len(x2),-1)
        q2 = evaluate(model,x2, batchsize=500)
        print("logp of generated data vs testing data:", torch.mean(q1),torch.mean(q2))
        fe_diffs = fe_diff_ntrials("../data/fe/",cfg,model,potential,nsamples)
        print(torch.mean(fe_diffs))
        print(torch.std(fe_diffs))


def fe_diff_ntrials(data_dir,cfg,model,potential, nsamples, relaxation=False,return_err=False):
    bar_list=[]
    for n in range(10):
        potential.update_data(data_dir+"run_%d/Fe_400K_test.xyz"%n) 
        bar, md, nf, c = fe_diff(cfg,model,potential, nsamples,relaxation=False, return_err=False)
        bar_list.append(bar)
    return torch.stack(bar_list)

if __name__ == "__main__":
    mode = sys.argv[2]
    main(mode=mode)