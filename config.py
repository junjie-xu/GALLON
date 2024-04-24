import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = CN()
    cfg.device = 'cuda:1'
    cfg.seed = [0,1,2,3,4]
    cfg.runs = 5  # Number of runs with random init
    cfg.gnn = CN()
    cfg.lm = CN()
    cfg.distill = CN()
    
    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset.name = 'hiv'
    cfg.dataset.target_task = 0   # Clintox: target_trask=1 if toxic; target_trask=0 if approved by FDA.
    cfg.dataset.split_method = 'random_scaffold' #['random', 'scaffold', 'random_scaffold']
    
    cfg.dataset.train_prop = 0.8
    cfg.dataset.val_prop = 0.1
    cfg.dataset.test_prop = 0.1

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    cfg.gnn.model.name = 'gcn'
    cfg.gnn.model.num_layers = 3
    cfg.gnn.model.hidden_dim = 32  # <----------------------------------------------------------------------------
    cfg.gnn.model.max_nodes = 132  # bbbp: 132   bace: 97   clintonx: 136

    cfg.gnn.train = CN()
    cfg.gnn.train.weight_decay = 0.0
    cfg.gnn.train.epochs = 800
    # cfg.gnn.train.early_stop = 50
    cfg.gnn.train.lr = 0.005
    cfg.gnn.train.wd = 0.0005  # weight_decay
    cfg.gnn.train.dropout = 0.3
    cfg.gnn.train.batch_size = 10000
    
    # ------------------------------------------------------------------------ #
    # Distill Model options
    # ------------------------------------------------------------------------ #
    cfg.distill.model = CN()
    cfg.distill.model.name = 'mlp'
    cfg.distill.model.num_layers = 3
    cfg.distill.model.hidden_dim = 32 # <----------------------------------------------------------------------------
    cfg.distill.model.max_nodes = 132 # bbbp: 132   bace: 97   clintonx: 136

    cfg.distill.train = CN()
    cfg.distill.train.weight_decay = 0.0
    cfg.distill.train.epochs = 1000
    cfg.distill.train.lr = 0.005
    cfg.distill.train.wd = 0.0005 # weight_decay
    cfg.distill.train.dropout = 0.3
    cfg.distill.train.batch_size = 45000
    cfg.distill.train.alpha = 0.1
    cfg.distill.train.beta = 0.1

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.name = 'roberta-base'
    cfg.lm.model.feat_shrink = ""

    cfg.lm.train = CN()
    cfg.lm.train.batch_size = 64
    cfg.lm.train.grad_acc_steps = 1   # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.lr = 2e-5
    cfg.lm.train.epochs = 10
    cfg.lm.train.warmup_epochs = 0.3
    cfg.lm.train.eval_patience = 5000 # Number of update steps between two evaluations
    cfg.lm.train.weight_decay = 0.00
    cfg.lm.train.dropout = 0.3   
    cfg.lm.train.att_dropout = 0.1  # The dropout ratio for the attention probabilities 
    cfg.lm.train.cla_dropout = 0.4  # The dropout ratio for the classifier 
    cfg.lm.train.diagram = True

    return cfg



def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())


