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
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--hidden_size', default=128, type=int, help='number of neurons in hidden layers')
parser.add_argument('--epochs', default=10, type=int, help='number of data-pass for training')
parser.add_argument('--device', default='0', type=str, help='visible GPU')

args = parser.parse_args()
test_set = ['jurkat', 'hepg2', 'k562', 'rpe1']
lr = args.lr
decay = args.decay
epochs = args.epochs
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

pert_data = PertData_Essential_v2()
pert_data.load()
gene2idx = get_gene_idx(pert_data)

signatures_dict = np.load('data/signatures_dict.npz', allow_pickle=True)
signatures_dict = signatures_dict['arr_0'].item()
signatures_list = np.load('data/signatures_list.npy')

for test_name in test_set[2:]:
  val_set = np.setdiff1d(test_set, [test_name])
  for val_name in val_set:
    pert_data.prepare_split(val=val_name, test=test_name)
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = 4*args.batch_size)
    print('+'*20+'test: '+test_name+', val: '+val_name+'+'*20)
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

        
        OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
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
        

        OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
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
            fname = 'predictions/'+pert_data.dataset_name+'_GEARS_lr='+str(lr)+'_decay='+str(decay)+'_hidden='+str(args.hidden_size)+'.npz'
            np.savez(fname, epoch=epoch, best_pearson_val=best_pearson_val, best_pred_val=best_pred_val, best_pred_test=best_pred_test, val_perturbs=val_perturbs, test_perturbs=test_perturbs)
        

        OE_preds = torch.from_numpy(pred2OE(preds.numpy(), signatures_dict, signatures_list, gene2idx))
        OE_truths = torch.from_numpy(pred2OE(truths.numpy(), signatures_dict, signatures_list, gene2idx))
        print('-'*30)
        metrics, aggregated_metrics = aggregated_eval_row(OE_preds, OE_truths, perts)
        print('OE-row-testing:', aggregated_metrics)
        metrics, aggregated_metrics = aggregated_eval_col(OE_preds, OE_truths)
        print('OE-col-testing:', aggregated_metrics)
        print('-'*30)
        
        
           
