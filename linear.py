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
parser.add_argument('--dataset', default='Adamson', type=str, help='the name for the dataset to use')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--method', default='genept', type=str, help='method to use (genept, scgpt, coexpress)')
parser.add_argument('--K', default=128, type=int, help='number of princeple componentse')



args = parser.parse_args()
SEED = [1,2,3,4,5]
decay = args.decay
reduced_GenePT = None
reduced_scGPT = None
if args.dataset in ['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential']:
    pert_data = PertData() 
    pert_data.load(data_name = args.dataset) 
elif args.dataset in ['jurkat', 'hepg2', 'k562', 'rpe1']:
    pert_data = PertData_Essential()
    pert_data.load(data_name=args.dataset)
gene2idx = get_gene_idx(pert_data)

OE_signatures = np.load('/oak/stanford/groups/ljerby/dzhu/Data/PertrubSeq_OE_signatures.npz', allow_pickle=True)
OE_signatures = OE_signatures['arr_0'].item()
column_names = np.load('/oak/stanford/groups/ljerby/dzhu/Data/PertrubSeq_GeneSetOE_Replogle2022_K562_column_names.npy')


for seed in SEED:
  set_all_seeds(seed)
  pert_data.prepare_split(split = 'simulation', seed = seed)
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
  print('+'*20+str(seed)+'+'*20)
  val_Y, val_V, val_perturbs = get_aggregated_data(pert_data,split='val')
  test_Y, test_V, test_perturbs = get_aggregated_data(pert_data,split='test')
  train_Y, train_V, train_perturbs = get_aggregated_data(pert_data,split='train')

  if args.method == 'genept' and reduced_GenePT is None:
    reduced_GenePT = get_extra_feat_dict_GenePT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)
  if args.method == 'scgpt' and reduced_scGPT is None:
    reduced_scGPT = get_extra_feat_dict_scGPT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)

  ctrl = np.array(pert_data.ctrl_mean)
  if args.dataset in ['k562', 'rpe1', 'jurkat', 'hepg2']:
    train_Y, val_Y, test_Y = train_Y.to(torch.float), val_Y.to(torch.float), test_Y.to(torch.float)
  else:
    ctrl_id = np.where(train_perturbs == 'ctrl')[0][0]
    ids_no_ctrl = np.setdiff1d(np.arange(train_Y.shape[0]),ctrl_id)
    train_Y, train_perturbs = train_Y[ids_no_ctrl], train_perturbs[ids_no_ctrl]
  Y = (train_Y-ctrl).to(torch.float)
  truths, perts = train_Y-ctrl, train_perturbs
  if args.method == 'genept':
    pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  elif args.method == 'scgpt':
    pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  if args.method in ['genept', 'scgpt']:
    pert = pert_emb #torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
    IW = torch.inverse(pert.t() @ pert + decay*torch.eye(pert.shape[-1])) 
    XY = pert.t() @ Y
    W = IW @ XY
    preds = pert @ W
  if args.method == 'coexpress':
    b = Y.mean(dim=0, keepdims=True)
    tmp = (Y-b)
    identity = tmp.t()@tmp
    eigenvalues, eigenvectors = eigsh(csc_matrix(identity.numpy()), k=args.K)
    V = torch.from_numpy(eigenvectors)
    G = V
    PG = pert2idx(gene2idx, perts, Y.shape[-1]) @ G
    W = torch.inverse(PG.t() @ PG + torch.eye(args.K)*decay) @ (PG.t() @ tmp @ G) @ torch.inverse(G.t() @ G + torch.eye(args.K)*decay)
    preds = (PG @ W) @ G.t() + b
    
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
    
  truths, perts = val_Y-ctrl, val_perturbs
  if args.method == 'genept':
    pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  elif args.method == 'scgpt':
    pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  if args.method in ['genept', 'scgpt']:
    pert = pert_emb #torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
    preds = pert @ W
  if args.method == 'coexpress':
    PG = pert2idx(gene2idx, perts, Y.shape[-1]) @ G
    preds = (PG @ W) @ G.t() + b
  best_pred_val = preds.detach().numpy()
  metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
  print('row-validation:', aggregated_metrics)
  metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
  print('col-validation:', aggregated_metrics)
  best_pearson_val = aggregated_metrics['pearson']

  OE_preds = torch.from_numpy(pred2OE(preds.numpy(), OE_signatures, column_names, gene2idx))
  OE_truths = torch.from_numpy(pred2OE(truths.numpy(), OE_signatures, column_names, gene2idx))
  print('-'*30)
  metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
  print('OE-row-validation:', aggregated_metrics)
  metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
  print('OE-col-validation:', aggregated_metrics)
  print('-'*30)

  truths, perts = test_Y-ctrl, test_perturbs
  if args.method == 'genept':
    pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  elif args.method == 'scgpt':
    pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
  if args.method in ['genept', 'scgpt']:
    pert = pert_emb #torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
    preds = pert @ W
  if args.method == 'coexpress':
    PG = pert2idx(gene2idx, perts, Y.shape[-1]) @ G
    preds = (PG @ W) @ G.t() + b 
  best_pred_test = preds.detach().numpy() 
  metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
  print('row-testing:', aggregated_metrics)
  metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
  print('col-testing:', aggregated_metrics)
  if args.method == 'coexpress':
    fname = 'predictions/'+pert_data.dataset_name+'_'+args.method+'_seed='+str(seed)+'_decay='+str(decay)+'_K='+str(args.K)+'.npz'
  else:
    fname = 'predictions/'+pert_data.dataset_name+'_'+args.method+'_seed='+str(seed)+'_decay='+str(decay)+'.npz'
  np.savez(fname, best_pearson_val=best_pearson_val, best_pred_val=best_pred_val, best_pred_test=best_pred_test, val_perturbs=val_perturbs, test_perturbs=test_perturbs)

  OE_preds = torch.from_numpy(pred2OE(preds.numpy(), OE_signatures, column_names, gene2idx))
  OE_truths = torch.from_numpy(pred2OE(truths.numpy(), OE_signatures, column_names, gene2idx))
  print('-'*30)
  metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
  print('OE-row-testing:', aggregated_metrics)
  metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
  print('OE-col-testing:', aggregated_metrics)
  print('-'*30)


 
