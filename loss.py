import torch
import numpy as np
import torch.nn as nn

eps = 1e-7


class GAR_mean(torch.nn.Module):
    def __init__(self, version='GAR', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.basic_loss = nn.L1Loss()
    def forward(self, y_pred, y_truth):
        return self.basic_loss(y_pred, y_truth)


def GAR_var(y_pred, y_truth, dim=0): # dim=0: column-wise; dim=1:row-wise
    pred_mean, truth_mean = y_pred.mean(dim=dim, keepdim=True), y_truth.mean(dim=dim, keepdim=True)
    diff = (y_pred-pred_mean) - (y_truth-truth_mean)
    loss_cov = diff**2/2
    return loss_cov.mean()

def GAR_pearson(y_pred, y_truth, dim=0): # dim=0: column-wise; dim=1:row-wise
    pred_mean, truth_mean = y_pred.mean(dim=dim, keepdim=True), y_truth.mean(dim=dim, keepdim=True)
    pred_std, truth_std = torch.clip(y_pred.std(dim=dim, keepdim=True), min=eps), torch.clip(y_truth.std(dim=dim, keepdim=True), min=eps)
    loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
    return loss_pearson.mean()

class GAR_row(torch.nn.Module):
    def __init__(self, alpha=1.0, version='GAR', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.alpha = alpha
        self.basic_loss = torch.nn.L1Loss(reduce=False)
        self.version = version

    def forward(self, y_pred, y_truth):
        pred_std, truth_std = torch.clip(y_pred.std(dim=1, keepdim=True), min=eps), torch.clip(y_truth.std(dim=1, keepdim=True), min=eps)
        pred_mean, truth_mean = y_pred.mean(dim=1, keepdim=True), y_truth.mean(dim=1, keepdim=True)
        loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        diff = (y_pred-pred_mean) - (y_truth-truth_mean)
        loss_cov = diff**2/2
        bloss = self.basic_loss(y_pred, y_truth).mean(dim=1, keepdim=True) + eps
        aloss = loss_pearson.mean(dim=1, keepdim=True) + eps
        closs = loss_cov.mean(dim=1, keepdim=True) + eps
        factor = torch.cat((bloss, aloss, closs), dim=1)
        if self.alpha > 1.0:
            factor = factor.min(dim=1, keepdim=True).values.detach()
        elif self.alpha < 1.0:
            factor = factor.max(dim=1, keepdim=True).values.detach()
        else:
            factor = 1.0
        aloss, bloss, closs = aloss/factor, bloss/factor, closs/factor
        loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha) + closs**(1/self.alpha))/3
        if self.version == 'GAR-EXP':
          loss = factor*(loss**self.alpha)
        else:
          loss = loss.log()*self.alpha
        return loss.mean()


class GAR_col(torch.nn.Module):
    def __init__(self, alpha=1.0, version='GAR', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.alpha = alpha
        self.basic_loss = torch.nn.L1Loss(reduce=False)
        self.version = version

    def forward(self, y_pred, y_truth):
        pred_std, truth_std = torch.clip(y_pred.std(dim=0, keepdim=True), min=eps), torch.clip(y_truth.std(dim=0, keepdim=True), min=eps)
        pred_mean, truth_mean = y_pred.mean(dim=0, keepdim=True), y_truth.mean(dim=0, keepdim=True)
        loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        diff = (y_pred-pred_mean) - (y_truth-truth_mean)
        loss_cov = diff**2/2
        bloss = self.basic_loss(y_pred, y_truth).mean(dim=0, keepdim=True) + eps
        aloss = loss_pearson.mean(dim=0, keepdim=True) + eps
        closs = loss_cov.mean(dim=0, keepdim=True) + eps
        factor = torch.cat((bloss, aloss, closs), dim=0)
        if self.alpha > 1.0:
            factor = factor.min(dim=0, keepdim=True).values.detach()
        elif self.alpha < 1.0:
            factor = factor.max(dim=0, keepdim=True).values.detach()
        else:
            factor = 1.0
        aloss, bloss, closs = aloss/factor, bloss/factor, closs/factor
        loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha) + closs**(1/self.alpha))/3
        if self.version == 'GAR-EXP':
          loss = factor*(loss**self.alpha)
        else:
          loss = loss.log()*self.alpha
        return loss.mean()

################################ original GAR function ############################################

class GAR(torch.nn.Module):
    def __init__(self, alpha=1.0, version='GAR', device = None):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.alpha = alpha
        self.basic_loss = nn.L1Loss()
        self.version = version

    def forward(self, y_pred, y_truth, alpha = None):
        if alpha is not None:
            self.alpha = alpha
        pred_std, truth_std = torch.clip(y_pred.std(axis=0), min=eps), torch.clip(y_truth.std(axis=0), min=eps)
        pred_mean, truth_mean = y_pred.mean(axis=0), y_truth.mean(axis=0)
        loss_pearson = ((y_pred-pred_mean)/pred_std - (y_truth-truth_mean)/truth_std)**2/2
        diff = (y_pred-pred_mean) - (y_truth-truth_mean)
        loss_cov = diff**2/2
        bloss = self.basic_loss(y_pred, y_truth)+eps
        aloss = loss_pearson.mean()+eps
        closs = loss_cov.mean()+eps
        if self.alpha > 1.0:
            factor = min([aloss, bloss, closs]).detach()
        elif self.alpha < 1.0:
            factor = max([aloss, bloss, closs]).detach()
        else:
            factor = 1.0
        aloss, bloss, closs = aloss/factor, bloss/factor, closs/factor
        loss = (aloss**(1/self.alpha) + bloss**(1/self.alpha) + closs**(1/self.alpha))/3
        if self.version == 'GAR-EXP':
          loss = factor*(loss**self.alpha)
        else:
          loss = loss.log()*self.alpha
        return loss 


################################ GEARS loss function ############################################

def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p!= 'ctrl' and dict_filter is not None:
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
        losses = losses + torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        ## direction loss
        if (p!= 'ctrl') and dict_filter is not None:
            losses = losses + torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            if ctrl.shape[0] == y.shape[0]:
                losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl[pert_idx]) -
                                                    torch.sign(pred_p - ctrl[pert_idx]))**2)/\
                                                    pred_p.shape[0]/pred_p.shape[1]
            else:
                losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                    torch.sign(pred_p - ctrl))**2)/\
                                                    pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))

