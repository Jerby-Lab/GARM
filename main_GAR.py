import torch
import numpy as np
from wrapper import *
from model import *
from loss import *
from utils import *
from pertdata import *
import argparse

parser = argparse.ArgumentParser(description = 'Perturb-Seq experiments')
parser.add_argument('--dataset', default='curated_k562', type=str, help='the name for the dataset to use')
parser.add_argument('--alpha', default=0.1, type=float, help='alpha for GAR loss')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--decay', default=1e-5, type=float, help='weight decay for training the model')
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--layers', default=1, type=int, help='layers of hidden neuronse')
parser.add_argument('--K', default=128, type=int, help='number of princeple componentse')
parser.add_argument('--verbose', default='True', type=str, help='whether to calculate and print all evals')
parser.add_argument('--iteration', default=10, type=int, help='number of inner loop iterations')
parser.add_argument('--epochs', default=300, type=int, help='number of outer loop iterations')

args = parser.parse_args()
SEED = [1,2,3,4,5]
lr = args.lr
decay = args.decay
batch_size = args.batch_size
layers = args.layers
epochs = args.epochs
milestones = [int(epochs/3), int(epochs/3*2)]
reduced_gene2feat = pickle.load(open('/oak/stanford/groups/ljerby/dzhu/Data/gene2GO_feature_D=256.pkl','rb'))
reduced_GenePT = None
DIM = 3072+256

if args.dataset in ['curated_k562', 'curated_rpe1']:
    # curated version of the data from GEARS paper
    pert_data = PertData_GEARS() 
    pert_data.load(data_name = args.dataset) 
elif args.dataset in ['jurkat', 'hepg2', 'k562', 'rpe1']:
    pert_data = PertData_Essential()
    pert_data.load(data_name=args.dataset)
gene2idx = get_gene_idx(pert_data)

signatures_dict = np.load('data/signatures_dict.npz', allow_pickle=True)
signatures_dict = signatures_dict['arr_0'].item()
signatures_list = np.load('data/signatures_list.npy')

for seed in SEED:
  set_all_seeds(seed)
  pert_data.prepare_split(split = 'simulation', seed = seed)
  pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
  print('+'*20+str(seed)+'+'*20)
  val_Y, val_V, val_perturbs = get_aggregated_data(pert_data,split='val')
  test_Y, test_V, test_perturbs = get_aggregated_data(pert_data,split='test')
  train_Y, train_V, train_perturbs = get_aggregated_data(pert_data,split='train')

  if reduced_GenePT is None: # only need do it once.
    reduced_GenePT = get_extra_feat_dict_GenePT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)
  ctrl = torch.from_numpy(np.array(pert_data.ctrl_mean))
  if args.dataset in ['k562', 'rpe1', 'jurkat', 'hepg2']:
    train_Y, val_Y, test_Y = train_Y.to(torch.float), val_Y.to(torch.float), test_Y.to(torch.float)
  else:
    ctrl_id = np.where(train_perturbs == 'ctrl')[0][0]
    ids_no_ctrl = np.setdiff1d(np.arange(train_Y.shape[0]),ctrl_id)
    train_Y, train_perturbs = train_Y[ids_no_ctrl], train_perturbs[ids_no_ctrl]
  truths, train_perts = train_Y-ctrl, train_perturbs
  model = Perturb_NN_GARM_tmp(D=truths.shape[-1], b=ctrl, hidden_sizes=([args.K]*layers), D_pert=3072, K=args.K)
  criterion1 = GAR_col(alpha=args.alpha)
  criterion2 = GAR_row(alpha=args.alpha)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
  best_pearson_val = -100
  best_pred_val = None
  best_pred_test = None
  for epoch in range(epochs):
    best_flag = False
    loss_value = 0
    y_all, pert_all = truths, train_perts
    N = y_all.shape[0]
    rand_ids = np.random.permutation(N)
    for i in range(0, N, batch_size):
      start = i
      end = i+batch_size
      if end >= N:
        end = N
      ids = rand_ids[start:end]
      y, pert = y_all[ids], pert_all[ids]
      pert_go = getPertEmb(pert, reduced_gene2feat) 
      pert_genept  = getPertEmb(pert, reduced_GenePT)
      pert = torch.cat([pert_go, pert_genept], dim=-1)
      pred_Y = model(pert_genept, pert_go)
      loss = (criterion1(pred_Y, y)+criterion2(pred_Y, y))/2
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
      optimizer.step()
      loss_value += loss.item()
    print('Epoch=%s, MSE=%.4f, lr=%.4f'%(epoch, loss_value, scheduler.get_last_lr()[0]))
    if (epoch+1) % 10 == 0:
      if args.verbose in ['True', 'true']:
        y, pert = truths, train_perturbs
        pert_go, pert_genept = getPertEmb(pert, reduced_gene2feat), getPertEmb(pert, reduced_GenePT)
        preds = model(pert_genept, pert_go)
        metrics, aggregated_metrics = aggregated_eval_row(preds, y, pert)
        print('row-train:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(preds, y)
        print('col-train:', aggregated_metrics)
          
        OE_preds = torch.from_numpy(pred2OE(preds.detach().numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(y.numpy(), signatures_dict, signatures_list, gene2idx))
        print('-'*30)
        metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, pert)
        print('OE-row-train:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
        print('OE-col-train:', aggregated_metrics)
        print('-'*30)
      
        y, pert = val_Y-ctrl, val_perturbs
        pert_go, pert_genept = getPertEmb(pert, reduced_gene2feat), getPertEmb(pert, reduced_GenePT)
        preds = model(pert_genept, pert_go)
        metrics, aggregated_metrics = aggregated_eval_row(preds, y, pert)
        print('row-validation:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(preds, y)
        print('col-validation:', aggregated_metrics)
        if aggregated_metrics['pearson'] > best_pearson_val:
            best_pearson_val = aggregated_metrics['pearson']
            best_pred_val = preds.detach().numpy()
            best_flag = True
  
        OE_preds = torch.from_numpy(pred2OE(preds.detach().numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(y.numpy(), signatures_dict, signatures_list, gene2idx))
        print('-'*30)
        metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, pert)
        print('OE-row-validation:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
        print('OE-col-validation:', aggregated_metrics)
        print('-'*30)
  
      y, pert = test_Y-ctrl, test_perturbs
      pert_go, pert_genept = getPertEmb(pert, reduced_gene2feat), getPertEmb(pert, reduced_GenePT)
      preds = model(pert_genept, pert_go)
      metrics, aggregated_metrics = aggregated_eval_row(preds, y, pert)
      print('row-testing:', aggregated_metrics)
      metrics, aggregated_metrics = aggregated_eval_col(preds, y)
      print('col-testing:', aggregated_metrics)
      if best_flag:
          best_pred_test = preds.detach().numpy()
          fname = 'predictions/'+pert_data.dataset_name+'_GAR_seed='+str(seed)+'_alpha='+str(args.alpha)+'_lr='+str(lr)+'_decay='+str(decay)+'_K='+str(args.K)+'_BS='+str(args.batch_size)+'.npz'
          np.savez(fname, epoch=epoch, best_pearson_val=best_pearson_val, best_pred_val=best_pred_val, best_pred_test=best_pred_test, val_perturbs=val_perturbs, test_perturbs=test_perturbs)
  
      if args.verbose in ['True','true']:
        OE_preds = torch.from_numpy(pred2OE(preds.detach().numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(y.numpy(), signatures_dict, signatures_list, gene2idx))
        print('-'*30)
        metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, pert)
        print('OE-row-testing:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
        print('OE-col-testing:', aggregated_metrics)
        print('-'*30)
              
          
    scheduler.step()
         
