from yacs.config import CfgNode as CN

cfg = CN()

cfg.device="cuda:0"

cfg.dataset = CN()
cfg.dataset.name = "Fe"
cfg.dataset.input_dir = "structures/lj.xyz"
cfg.dataset.nparticles = 32
cfg.dataset.kT = float(1)
cfg.dataset.rho = None
cfg.dataset.ncellx = None
cfg.dataset.ncelly = None
cfg.dataset.ncellz = None
cfg.dataset.cell_len = None

cfg.flow=CN()
cfg.flow.type= "NSF_CL"
cfg.flow.nlayers=3
cfg.flow.nsplines =32
cfg.flow.hidden_dim = 100

cfg.prior=CN()
cfg.prior.type = "lattice"
cfg.prior.lattice_dir = "structures/ref.xyz"
cfg.prior.alpha = 100

cfg.train_parameters = CN()
cfg.train_parameters.max_epochs = 4000
cfg.train_parameters.batch_size = 100
cfg.train_parameters.output_freq = 100
cfg.train_parameters.learning_rate = 1e-4
cfg.train_parameters.lr_scheduler_gamma = 0.999

cfg.output=CN()
cfg.output.pos_dir="./positions_during_training.xyz"
cfg.output.model_dir="saved_models/"

def get_cfg_defaults():
  return cfg.clone()
