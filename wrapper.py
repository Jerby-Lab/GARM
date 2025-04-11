from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import GEARS_Model, GEARS_Model_GAR_M
from inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
from utils import get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction
from loss import *


import warnings
warnings.filterwarnings("ignore")


class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.ctrl_adata = pert_data.ctrl_adata
        self.ctrl_mean = torch.from_numpy(pert_data.ctrl_mean).view(-1).to(self.device)
        self.saved_pred = {}
        self.saved_logvar_sum = {}

        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         lr = 1e-3,
                         weight_decay = 5e-4,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        self.best_model = deepcopy(self.model)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.min_val = -np.inf


    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, mode='val'): # if not trained in single cell
        loader_name = mode+'_loader'
        truths = []
        preds = []
        perts = []
        for step, batch in enumerate(self.dataloader[loader_name]):
            batch.to(self.device)
            ctrl = batch.x.view(batch.y.shape)
            pred = self.model(batch)
            preds.append((pred-ctrl).detach().cpu()) 
            truths.append((batch.y-ctrl).cpu()) 
            perts.extend(batch.pert) 
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        return preds, truths, perts

    
    def predict_list(self, pert_list, mode='model'): # model or best_model based on pearson
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
        
        if mode == 'model':
            model = self.model.to(self.device)
        elif mode == 'best_model':
            model = self.best_model.to(self.device)
        model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        for pert in pert_list:            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata, self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = model(batch)
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred

    
    def train(self, epochs = 20):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        
        self.model = self.model.to(self.device)
        optimizer = self.optimizer 
        scheduler = self.scheduler 

        print_sys('Start Training...')
        for epoch in range(epochs):
            self.model.train()
            tr_loss = []
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                pert = batch.pert
                y = batch.y
                ctrl = batch.x.view(y.shape)
                if self.ctrl_mean is not None:
                    ctrl = self.ctrl_mean
                optimizer.zero_grad()
                pred = self.model(batch)
                loss = loss_fct(pred, y, pert,
                              ctrl = ctrl, 
                              dict_filter = self.dict_filter,
                              direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()
                tr_loss.append(loss.item())
            if epoch % 20 == 0:
                log = "Epoch {} Step {} Train Loss: {:.4f}" 
                print(log.format(epoch, step, np.mean(tr_loss)))

            scheduler.step()           
        print_sys("Done and skip GEARS evaluation!")
        
        return



class GEARS_GAR_M:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 pert_aggregation = False,
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        
        self.gene_list = pert_data.gene_names.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        self.pert_aggregation = pert_aggregation
        
        
        try:
            self.adata = None
            self.set2conditions = None
            self.train_Y = pert_data.train_Y
            self.ctrl_mean = None
            self.ctrl_adata = None
        except:
            self.adata = pert_data.adata
            self.set2conditions = pert_data.set2conditions
            self.train_Y = None
            self.ctrl_mean = torch.from_numpy(pert_data.ctrl_mean).to(self.device)
            self.ctrl_adata = pert_data.ctrl_adata
        
        self.dict_filter = None
        

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         lr = 1e-3,
                         weight_decay = 5e-4,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               train_Y=self.train_Y,
                                               gene_list=self.gene_list)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model_GAR_M(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        self.best_model = deepcopy(self.model)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.min_val = -np.inf
        self.GAR_mean = GAR_mean()


    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))

    def predict_list(self, pert_list): 
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        model = self.model.to(self.device)
        model.eval()
        results_pred = {}
        from torch_geometric.data import DataLoader
        p_mean, p_var_row, p_pear_row, p_var_col, p_pear_col = [],[],[],[],[]
        for pert in pert_list:            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device, num_samples=10)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = model(batch)        
            p_mean.append(pred_mean.mean(dim=0))
            p_var_row.append(pred_var_row.mean(dim=0))
            p_pear_row.append(pred_pear_row.mean(dim=0))
            p_var_col.append(pred_var_col.mean(dim=0))
            p_pear_col.append(pred_pear_col.mean(dim=0))
        p_mean = torch.stack(p_mean) 
        p_var_row, p_pear_row = torch.stack(p_var_row), torch.stack(p_pear_row) 
        p_var_col, p_pear_col = torch.stack(p_var_col), torch.stack(p_pear_col)
        preds,_,_ = model.predict(p_mean-self.ctrl_mean, p_var_row-self.ctrl_mean, p_pear_row-self.ctrl_mean, p_var_col-self.ctrl_mean, p_pear_col-self.ctrl_mean)
        preds = preds + self.ctrl_mean
        for i in range(len(pert_list)):
            pert = pert_list[i]
            p = preds[i]
            results_pred['_'.join(pert)] = p.detach().cpu().numpy()
                
        self.saved_pred.update(results_pred)
        return results_pred


    
    def predict(self, mode='val'): # model or best_model based on pearson
        loader_name = mode+'_loader'
        truths = []
        preds = []
        perts = []
        for step, batch in enumerate(self.dataloader[loader_name]):
            batch.to(self.device)
            ctrl = batch.x.view(batch.y.shape)
            pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = self.model(batch)
            pred, pred_row, pred_col = self.model.predict(pred_mean-ctrl, pred_var_row-ctrl, pred_pear_row-ctrl, pred_var_col-ctrl, pred_pear_col-ctrl)
            pred, pred_row, pred_col = pred+ctrl, pred_row+ctrl, pred_col+ctrl
            preds.append((pred-ctrl).detach().cpu()) 
            truths.append((batch.y-ctrl).cpu()) 
            perts.extend(batch.pert) 
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        return preds, truths, perts
    
    def train(self, epochs = 20):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """

        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        
        self.model = self.model.to(self.device)
        optimizer = self.optimizer
        scheduler = self.scheduler

        print_sys('Start Training...')
        for epoch in range(epochs):
            self.model.train()
            tr_loss = []
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                pert = batch.pert
                y = batch.y
                ctrl = batch.x.view(y.shape)
                optimizer.zero_grad()
                pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col = self.model(batch)
                pred_Y, pred_Y_row, pred_Y_col = self.model.predict(pred_mean-ctrl, pred_var_row-ctrl, pred_pear_row-ctrl, pred_var_col-ctrl, pred_pear_col-ctrl)
                pred_Y, pred_Y_row, pred_Y_col = pred_Y+ctrl, pred_Y_row+ctrl, pred_Y_col+ctrl
                loss_mean = self.GAR_mean(pred_mean-ctrl, y-ctrl)
                loss_var_row = GAR_var(pred_var_row - ctrl, y - ctrl, dim=1)
                loss_pearson_row = GAR_pearson(pred_pear_row - ctrl, y - ctrl, dim=1)
                loss_var_col = GAR_var(pred_var_row - ctrl, y - ctrl, dim=0)
                loss_pearson_col = GAR_pearson(pred_pear_row - ctrl, y - ctrl, dim=0)
                loss_final = self.GAR_mean(pred_Y-ctrl, y-ctrl) 
                loss_final = loss_final + self.GAR_mean(pred_Y_row-ctrl, y-ctrl) + self.GAR_mean(pred_Y_col-ctrl, y-ctrl)
                loss = loss_mean+loss_var_row+loss_pearson_row+loss_var_col+loss_pearson_col+loss_final
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                #if self.wandb:
                #    self.wandb.log({'training_loss': loss.item()})
                tr_loss.append(loss.item())
            if epoch % 20 == 0:
                log = "Epoch {} Step {} Train Loss: {:.4f}" 
                print(log.format(epoch, step, np.mean(tr_loss)))

            scheduler.step()

        return
