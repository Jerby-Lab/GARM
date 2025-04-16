import torch
import numpy as np
from wrapper import *
from dataset import *
from model import *
from loss import *
from utils import *
from pertdata import *
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix
import argparse
import os

parser = argparse.ArgumentParser(description = 'Perturb-Seq experiments')
parser.add_argument('--dataset', default='adamson', type=str, help='the name for the dataset to use')
parser.add_argument('--method', default='GARM', type=str, help='method to use')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay for training the model')
parser.add_argument('--hidden_size', default=128, type=int, help='GNN hidden size')

args = parser.parse_args()
if args.method in ['GEARS']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = [1,2,3,4,5]
lr = args.lr
decay = args.decay
epochs = 10
if args.dataset in ['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential']:
    pert_data = PertData() 
    pert_data.load(data_name = args.dataset) 
elif args.dataset in ['jurkat', 'hepg2', 'k562', 'rpe1']:
    pert_data = PertData_Essential()
    pert_data.load(data_name=args.dataset)
gene2idx = get_gene_idx(pert_data)

OE_signatures = np.load('/oak/stanford/groups/ljerby/dzhu/Data/PertrubSeq_OE_signatures.npz', allow_pickle=True)
OE_signatures = OE_signatures['arr_0'].item()
OE_truth = []
row_names = []
column_names = np.load('/oak/stanford/groups/ljerby/dzhu/Data/PertrubSeq_GeneSetOE_Replogle2022_K562_column_names.npy')
for k in pert_data.dataset_processed.keys():
    i = pert_data.dataset_processed[k][0]
    row_names.append(i.pert)
    OE_truth.append(i.y)
OE_truth = torch.cat(OE_truth, dim=0)
OE_truth = pred2OE(OE_truth.numpy(), OE_signatures, column_names, gene2idx)
pert2OEidx = {row_names[i]:i for i in range(len(row_names))}
OE_truth_ctrl = pred2OE(np.array(pert_data.ctrl_mean), OE_signatures, column_names, gene2idx)


for seed in SEED:
  set_all_seeds(seed)
  pert_data.prepare_split(split = 'simulation', seed = seed)
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
  print('+'*20+str(seed)+'+'*20)
  if args.method == 'GARM':
      gears_model = GEARS_GAR_M(pert_data, device = 'cuda:0', 
                                weight_bias_track = False,
                                proj_name = 'pertnet', 
                                exp_name = 'pertnet')
  elif args.method == 'GEARS':
      gears_model = GEARS(pert_data, device = 'cuda:0', 
                                weight_bias_track = False,
                                proj_name = 'pertnet', 
                                exp_name = 'pertnet')
  gears_model.model_initialize(hidden_size = args.hidden_size, lr=lr, weight_decay=decay)
  gears_model.scheduler = torch.optim.lr_scheduler.StepLR(gears_model.optimizer, step_size=1, gamma=0.997)
  gears_model.tunable_parameters()
  best_pearson_val = -100
  best_pred_val = None
  best_pred_test = None
  for epoch in range(epochs):
    best_flag = False
    gears_model.train(epochs = 50)
    print('Epoch=%s, lr=%.4f'%(epoch, gears_model.scheduler.get_last_lr()[0]))
    # evaluations
    preds, truths, perts = gears_model.predict(mode='train')
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-train:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-train:', aggregated_metrics)
    
    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), OE_signatures, column_names, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), OE_signatures, column_names, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-train:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-train:', aggregated_metrics)
    print('-'*30)


    preds, truths, perts = gears_model.predict(mode='val')
    val_perturbs = perts
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-validation:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-validation:', aggregated_metrics)
    if aggregated_metrics['pearson'] > best_pearson_val:
        best_pearson_val = aggregated_metrics['pearson']
        best_pred_val = preds
        best_flag = True
    
    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), OE_signatures, column_names, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), OE_signatures, column_names, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-validation:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-validation:', aggregated_metrics)
    print('-'*30)



    preds, truths, perts = gears_model.predict(mode='test')
    test_perturbs = perts
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-testing:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-testing:', aggregated_metrics)
    if best_flag:
        best_pred_test = preds
        fname = 'predictions/'+pert_data.dataset_name+'_'+args.method+'_seed='+str(seed)+'_lr='+str(lr)+'_decay='+str(decay)+'_hidden='+str(args.hidden_size)+'.npz'
        np.savez(fname, epoch=epoch, best_pearson_val=best_pearson_val, best_pred_val=best_pred_val, best_pred_test=best_pred_test, val_perturbs=val_perturbs, test_perturbs=test_perturbs)
    
    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), OE_signatures, column_names, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), OE_signatures, column_names, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-testing:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-testing:', aggregated_metrics)
    print('-'*30)   
 
