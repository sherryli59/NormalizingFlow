from yacs.config import CfgNode as CN

cfg = CN()

cfg.device="cuda:0"

cfg.dataset = CN()
cfg.dataset.name = None
cfg.dataset.potential = None
cfg.dataset.training_data = None
cfg.dataset.testing_data = None
cfg.dataset.data = None
cfg.dataset.nparticles = 32
cfg.dataset.dim = 3
cfg.dataset.kT = float(1)
cfg.dataset.rho = None
cfg.dataset.ncellx = None
cfg.dataset.ncelly = None
cfg.dataset.ncellz = None
cfg.dataset.cell_len = None
cfg.dataset.boxlength = None
cfg.dataset.periodic = True
cfg.dataset.type = "xyz"
#LJ
cfg.dataset.sigma = 1.0
cfg.dataset.epsilon = 1.0
cfg.dataset.cutoff = 1.6
cfg.dataset.shift = True
#GaussianMixture/EinsteinCrystal
cfg.dataset.centers = None
cfg.dataset.vars = None
cfg.dataset.alpha = None
#Fe
cfg.dataset.input_dir = None

cfg.flow=CN()
cfg.flow.type= "NSF_AR"
cfg.flow.nlayers=3
cfg.flow.nsplines =32
cfg.flow.hidden_dim = 100

cfg.prior=CN()
cfg.prior.type = None
cfg.prior.lattice_dir = None
cfg.prior.alpha = 100
cfg.prior.centers = None
cfg.prior.vars = None
cfg.prior.nparticles = cfg.dataset.nparticles 
cfg.prior.dim = cfg.dataset.dim
cfg.prior.boxlength = cfg.dataset.boxlength


cfg.train_parameters = CN()
cfg.train_parameters.max_epochs = 4000
cfg.train_parameters.batch_size = 100
cfg.train_parameters.output_freq = 100
cfg.train_parameters.learning_rate = 1e-4
cfg.train_parameters.scheduler = "exponential"
cfg.train_parameters.lr_scheduler_gamma = 0.999

cfg.output=CN()
cfg.output.training_dir="../training/"
cfg.output.testing_dir="../testing/"
cfg.output.model_dir="../saved_models/"
cfg.output.best_model_dir = "../trained_models/"


def get_cfg_defaults():
  return cfg.clone()
