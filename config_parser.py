def parser_add_main_args(parser):
    # Basic Options
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1, 2, 3, 4])
    
    # Dataset
    parser.add_argument('--dataname', type=str, default='freesolv')
    parser.add_argument('--split_method', type=str, default='random_scaffold', 
                        choices=['random', 'scaffold', 'random_scaffold'])
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--test_prop', type=float, default=0.1)
    
    # GNN
    parser.add_argument('--gnn_name', type=str, default='gcn')
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--gnn_hidden', type=int, default=32)
    parser.add_argument('--gnn_max_nodes', type=int, default=132) # bbbp: 132   bace: 97   clintonx: 136
    
    parser.add_argument('--gnn_wd', type=float, default=0.0)
    parser.add_argument('--gnn_epochs', type=int, default=500)
    parser.add_argument('--gnn_lr', type=float, default=0.005)
    parser.add_argument('--gnn_dropout', type=float, default=0.3)
    parser.add_argument('--gnn_batch_size', type=int, default=10000)
    
    # Distill
    parser.add_argument('--distill_name', type=str, default='mlp')
    parser.add_argument('--distill_layers', type=int, default=3)
    parser.add_argument('--distill_hidden', type=int, default=32)
    parser.add_argument('--distill_max_nodes', type=int, default=132) # bbbp: 132   bace: 97   clintonx: 136
    
    parser.add_argument('--distill_wd', type=float, default=0.0)
    parser.add_argument('--distill_epochs', type=int, default=500)
    parser.add_argument('--distill_lr', type=float, default=0.005)
    parser.add_argument('--distill_dropout', type=float, default=0.3)
    parser.add_argument('--distill_batch_size', type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    
    # LM
    parser.add_argument('--lm_name', type=str, default='roberta-base')
    parser.add_argument('--feat_shrink', type=str, default='')
    
    parser.add_argument('--lm_batch_size', type=int, default=40)
    parser.add_argument('--lm_grad_acc_steps', type=int, default=1)
    parser.add_argument('--lm_lr', type=float, default=0.005)
    parser.add_argument('--lm_epochs', type=int, default=10)
    parser.add_argument('--lm_warmup_epochs', type=float, default=0.3)
    parser.add_argument('--lm_eval_patience', type=int, default=5000)
    parser.add_argument('--lm_wd', type=float, default=0.0)
    parser.add_argument('--lm_att_dropout', type=float, default=0.1)
    parser.add_argument('--lm_cla_dropout', type=float, default=0.4)
    
