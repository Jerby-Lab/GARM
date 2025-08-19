import torch
import numpy as np
from wrapper import *
from model import *
from loss import *
from utils import *
from pertdata import *
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix
import argparse
import os

parser = argparse.ArgumentParser(description = 'Perturb-Seq experiments')
parser.add_argument('--dataset', default='nadig-replogle', type=str, help='the name for the dataset to use')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--method', default='genept', type=str, help='method to use (genept, scgpt, coexpress)')
parser.add_argument('--K', default=128, type=int, help='number of princeple componentse')



args = parser.parse_args()
test_set = ['jurkat', 'hepg2', 'k562', 'rpe1']
decay = args.decay
reduced_GenePT = None
reduced_scGPT = None
pert_data = PertData_Essential_v2()
pert_data.load()
gene2idx = get_gene_idx(pert_data)

signatures_dict = np.load('data/signatures_dict.npz', allow_pickle=True)
signatures_dict = signatures_dict['arr_0'].item()
signatures_list = np.load('data/signatures_list.npy')

for test_name in test_set:
  val_set = np.setdiff1d(test_set, [test_name])
  for val_name in val_set:
    pert_data.prepare_split(val=val_name, test=test_name)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    print('+'*20+'test: '+test_name+', val: '+val_name+'+'*20)
    val_Y, val_V, val_perturbs = get_aggregated_data(pert_data,split='val', delta=True)
    test_Y, test_V, test_perturbs = get_aggregated_data(pert_data,split='test', delta=True)
    train_Y, train_V, train_perturbs = get_aggregated_data(pert_data,split='train', delta=True)

    if args.method == 'genept' and reduced_GenePT is None:
      reduced_GenePT = get_extra_feat_dict_GenePT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)
    if args.method == 'scgpt' and reduced_scGPT is None:
      reduced_scGPT = get_extra_feat_dict_scGPT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)

    train_Y, val_Y, test_Y = train_Y.to(torch.float), val_Y.to(torch.float), test_Y.to(torch.float)
    Y = train_Y
    truths, perts = train_Y, train_perturbs
    if args.method == 'genept':
      pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    elif args.method == 'scgpt':
      pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    if args.method in ['genept', 'scgpt']:
      pert = torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
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
    if args.method == 'non-ctrl-mean':
      b = Y.mean(dim=0, keepdims=True)
      preds = b.repeat(truths.shape[0], 1)
      
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-train:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-train:', aggregated_metrics)

    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-train:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-train:', aggregated_metrics)
    print('-'*30)
      
    truths, perts = val_Y, val_perturbs
    if args.method == 'genept':
      pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    elif args.method == 'scgpt':
      pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    if args.method in ['genept', 'scgpt']:
      pert = torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
      preds = pert @ W
    if args.method == 'coexpress':
      PG = pert2idx(gene2idx, perts, Y.shape[-1]) @ G
      preds = (PG @ W) @ G.t() + b
    if args.method == 'non-ctrl-mean':
      preds = b.repeat(truths.shape[0], 1)
      
    best_pred_val = preds.detach().numpy()
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-validation:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-validation:', aggregated_metrics)
    best_pearson_val = aggregated_metrics['pearson']

    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-validation:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-validation:', aggregated_metrics)
    print('-'*30)

    truths, perts = test_Y, test_perturbs
    if args.method == 'genept':
      pert_emb = getPertEmb(perts, reduced_GenePT) #torch.from_numpy(np.stack([reduced_GenePT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    elif args.method == 'scgpt':
      pert_emb = getPertEmb(perts, reduced_scGPT) #torch.from_numpy(np.stack([reduced_scGPT[pt.split('+')[0]] for pt in perts])).to(torch.float)
    if args.method in ['genept', 'scgpt']:
      pert = torch.nn.functional.normalize(pert_emb, dim=1) #torch.cat([pert_emb, torch.ones(pert_emb.shape[0], 1)], dim=-1)
      preds = pert @ W
    if args.method == 'coexpress':
      PG = pert2idx(gene2idx, perts, Y.shape[-1]) @ G
      preds = (PG @ W) @ G.t() + b
    if args.method == 'non-ctrl-mean':
      preds = b.repeat(truths.shape[0], 1)
    
    best_pred_test = preds.detach().numpy() 
    metrics, aggregated_metrics = aggregated_eval_row(preds, truths, perts)
    print('row-testing:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(preds, truths)
    print('col-testing:', aggregated_metrics)
    if args.method == 'coexpress':
      fname = 'predictions/'+pert_data.dataset_name+'_'+args.method+'_decay='+str(decay)+'_K='+str(args.K)+'.npz'
    else:
      fname = 'predictions/'+pert_data.dataset_name+'_'+args.method+'_decay='+str(decay)+'.npz'
    np.savez(fname, best_pearson_val=best_pearson_val, best_pred_val=best_pred_val, best_pred_test=best_pred_test, val_perturbs=val_perturbs, test_perturbs=test_perturbs)

    OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
    OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
    print('-'*30)
    metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
    print('OE-row-testing:', aggregated_metrics)
    metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
    print('OE-col-testing:', aggregated_metrics)
    print('-'*30)


   
