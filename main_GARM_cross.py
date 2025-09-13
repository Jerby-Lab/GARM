import torch
import numpy as np
from wrapper import *
from model import *
from loss import *
from utils import *
from pertdata import *
import argparse

parser = argparse.ArgumentParser(description = 'Perturb-Seq experiments')
parser.add_argument('--dataset', default='nadig-replogle', type=str, help='the name for the dataset to use')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--decay', default=1e-6, type=float, help='weight decay for training the model')
parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
parser.add_argument('--layers', default=2, type=int, help='number of layers for FFNN')
parser.add_argument('--K', default=1024, type=int, help='number of neurons in hidden layers')
parser.add_argument('--verbose', default='True', type=str, help='whether to calculate and print all evals and store out predictions')
parser.add_argument('--epochs', default=300, type=int, help='number of data-pass for training')

args = parser.parse_args()
test_set = ['jurkat', 'hepg2', 'k562', 'rpe1']
lr = args.lr
decay = args.decay
batch_size = args.batch_size
layers = args.layers
epochs = args.epochs
milestones = [int(epochs/3), int(epochs/3*2)]
reduced_gene2feat = pickle.load(open('/oak/stanford/groups/ljerby/dzhu/Data/gene2GO_feature_D=256.pkl','rb'))
reduced_GenePT = None
DIM = 3072+256

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
    val_Y, val_V, val_perturbs = get_aggregated_data(pert_data,split='val',delta=True)
    test_Y, test_V, test_perturbs = get_aggregated_data(pert_data,split='test',delta=True)
    train_Y, train_V, train_perturbs = get_aggregated_data(pert_data,split='train',delta=True)

    if reduced_GenePT is None: # only need do it once.
      reduced_GenePT = get_extra_feat_dict_GenePT(get_genes_from_perts(np.concatenate([train_perturbs, val_perturbs, test_perturbs], axis=0)), k=-1)
    train_Y, val_Y, test_Y = train_Y.to(torch.float), val_Y.to(torch.float), test_Y.to(torch.float)
    truths, train_perts = train_Y, train_perturbs
    model = Perturb_NN_GARM(D=truths.shape[-1], hidden_sizes=([args.K]*layers), D_pert=3072, K=args.K)
    loss_mean = GAR_mean()
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
        pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = model(pert_genept, pert_go)
        pred_Y, pred_Y_row, pred_Y_col = model.predict(pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col)
        l_mean = loss_mean(pred_mean, y)
        l_var_row = GAR_var(pred_var_row, y, dim=1)
        l_pearson_row = GAR_pearson(pred_pear_row, y, dim=1)
        l_var_col = GAR_var(pred_var_col, y, dim=0)
        l_pearson_col = GAR_pearson(pred_pear_col, y, dim=0)
        loss_final = loss_mean(pred_Y, y) 
        loss_final = loss_final + loss_mean(pred_Y_row, y) + loss_mean(pred_Y_col, y)
        loss = l_mean+l_var_row+l_pearson_row+l_var_col+l_pearson_col+loss_final
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
          pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = model(pert_genept, pert_go)
          preds,_,_ = model.predict(pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col)
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
        
          y, pert = val_Y, val_perturbs
          pert_go, pert_genept = getPertEmb(pert, reduced_gene2feat), getPertEmb(pert, reduced_GenePT)
          pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = model(pert_genept, pert_go)
          preds,_,_ = model.predict(pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col)
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

        y, pert = test_Y, test_perturbs
        pert_go, pert_genept = getPertEmb(pert, reduced_gene2feat), getPertEmb(pert, reduced_GenePT)
        pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = model(pert_genept, pert_go)
        preds,_,_ = model.predict(pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col)
        metrics, aggregated_metrics = aggregated_eval_row(preds, y, pert)
        print('row-testing:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(preds, y)
        print('col-testing:', aggregated_metrics)
        if best_flag:
            best_pred_test = preds.detach().numpy()
            fname = 'predictions/'+pert_data.dataset_name+'_GARM_lr='+str(lr)+'_decay='+str(decay)+'_K='+str(args.K)+'_BS='+str(args.batch_size)+'.npz'
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
           
