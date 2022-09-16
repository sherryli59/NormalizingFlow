import torch
import sys
sys.path.append("../../")
sys.path.append("../")
from src import setup,utils,dynamics



def train(cfg,model,optimizer,scheduler,logger,potential,sim=None):
    losses=[]
    max_logprob=torch.log(torch.tensor(0))
    for i in range(cfg.train_parameters.max_epochs):
        optimizer.zero_grad()
        if (sim is not None) and (i % (2*cfg.train_parameters.output_freq) == 0):
            pos, acc_prob = dynamics.collect_hmc_data(cfg,model,sim)
            if (acc_prob>0.3) and (acc_prob<0.6):
                x = utils.subsample(pos,cfg.train_parameters.batch_size).to(cfg.device)
            else: 
                x = potential.sample(cfg.train_parameters.batch_size,flatten=True).to(cfg.device)
        else:
            x = potential.sample(cfg.train_parameters.batch_size,flatten=True).to(cfg.device)
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        forward_loss=-torch.mean(logprob)
        loss = forward_loss
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
            
            if (i>0) and (-forward_loss>max_logprob):
                max_logprob=-forward_loss
                torch.save({"model":model.state_dict(),"optim": optimizer.state_dict(),"scheduler": scheduler.state_dict(),"epoch": i+1,
                            "loss":losses},cfg.output.model_dir+cfg.dataset.name+'%d.pth'% (i//cfg.train_parameters.output_freq))
def main():
    name=sys.argv[1]
    cfg=setup.read_input("../input/%s.yaml"%name)
    model,optimizer,scheduler,logger,potential = setup.setup_model(cfg)
    train(cfg,model,optimizer,scheduler,logger,potential)
    
    
if __name__ == "__main__":
    main()
