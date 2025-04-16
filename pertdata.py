from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter,\
                  print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper,\
                  np_pearson_cor, set_all_seeds



class PertData_Essential_v2:
    def __init__(self, data_path= '/oak/stanford/groups/ljerby/dzhu/Data',
                 gene_set_path=None, 
                 default_pert_graph=True):
        """
        For Cross-Data Predictions.
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """

        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = data_path
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.ctrl_mean = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = 'simulation'
        self.seed = 1
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['gene'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path, 'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self):
        # load the Pert x Gene data for Jurkat, Hepg2, K562 and RPE1, where only overlapped genes are included.
        data_path = os.path.join(self.data_path, 'shared_essential_graphs.pkl')
        self.dataset_processed = pickle.load(open(data_path,'rb'))
        self.condition_names = []
        for name in self.dataset_processed.keys():
            self.condition_names.extend(list(self.dataset_processed[name].keys()))
        self.condition_names = np.unique(self.condition_names)
        print('Unique perturbation number:', self.condition_names.shape)
        self.set_pert_genes()
        print('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        self.not_in_go_pert = [x for x in self.condition_names if x not in self.pert_names]
        print(len(self.not_in_go_pert), self.not_in_go_pert)
        
        self.in_go_pert = [x for x in self.condition_names if x in self.pert_names]
        # load the gene names for the overlapped genes
        self.gene_names = np.load('/oak/stanford/groups/ljerby/dzhu/Data/weissman_shared_genes.npy', allow_pickle=True)
        self.node_map = {x: it for it, x in enumerate(self.gene_names)}

        print_sys("Done!")


    def create_train_matrix(self):
        """
        Set the train matrix for generate gene co-expression graph.
        """

        train_Y = []
        for name in self.split_data['train']:
            for p in self.dataset_processed[name].keys():
                train_Y.append(self.dataset_processed[name][p][0].y)
        self.train_Y = torch.cat(train_Y, dim=0)
        

    def prepare_split(self, val='hepg2', test='jurkat'):
        """
        One for testing, one for validation, and remaining for training
        """

        self.dataset_name = 'val:'+val+',test:'+test
        datasets = ['k562','rpe1','hepg2','jurkat']
        self.split_data = {'train':[], 'val':[], 'test':[]}
        self.split_data['train'] = np.setdiff1d(datasets, [val, test])
        self.split_data['val'] = [val]
        self.split_data['test'] = [test]
        print_sys("Creating training matrix")
        self.create_train_matrix()


    def tpm2OE_avg(self, matOE=None):
        """
        Aggregate OE (avg) expression with matOE. This function mainly used for directly training on OE level.
        """

        if matOE is not None:
            for name in self.dataset_processed.keys():
                for p in self.dataset_processed[name].keys():
                    for i in range(len(self.dataset_processed[name][p])):
                        self.dataset_processed[name][p][i].y = self.dataset_processed[name][p][i].y @ matOE
                        self.dataset_processed[name][p][i].x = self.dataset_processed[name][p][i].x.T @ matOE


        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            

        # Create cell graphs
        cell_graphs = {}
        if True:
            splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for name in self.split_data[i]:
                    for p in self.dataset_processed[name].keys():
                        if p not in self.not_in_go_pert:
                            cell_graphs[i].extend(self.dataset_processed[name][p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True) #, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
            self.dataloader =  {'train_loader': train_loader,
                                'val_loader': val_loader,
                                'test_loader': test_loader}

            print_sys("Done!")







class PertData_Essential:
    def __init__(self, data_path= '/oak/stanford/groups/ljerby/dzhu/Data',
                 gene_set_path=None, 
                 default_pert_graph=True):
        """
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """

        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = 'simulation'
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None
        self.avg_sc = True
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['gene'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, data_name = 'jurkat', avg_sc=True, subset=False):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path

        Parameters
        ----------
        data_name: str
            Name of dataset

        Returns
        -------
        None

        """
        if data_name in ['jurkat', 'hepg2', 'k562', 'rpe1']:
            data_path = os.path.join(self.data_path, data_name+'_essential_sc')
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, data_name+'_essential_raw_singlecell.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            lengths = np.array(self.adata.var['length'])
            self.adata.X = (self.adata.X / lengths)
            self.adata.X = self.adata.X/self.adata.X.sum(axis=-1,keepdims=True)
            self.adata.X = self.adata.X*1e5
            self.adata.X = np.log2(self.adata.X+1)
            if subset:
                ctrl_adata = self.adata[self.adata.obs['gene'] == 'non-targeting']
                ctrl_expressed = ctrl_adata.X > 0
                ctrl_expressed_ct = ctrl_expressed.sum(axis=0)
                ctrl_N = ctrl_adata.X.shape[0]
                ctrl_expressed_ratio = ctrl_expressed_ct/ctrl_N 
                selected_gene_ids = np.where(ctrl_expressed_ratio > 0.2)[0]
                print(self.adata)
                print('subset:',selected_gene_ids.shape[0], 'over', self.adata.X.shape[1])
                aggregated_tpm=[]
                for ptb in np.unique(self.adata.obs['gene']):
                    aggregated_tpm.append(self.adata.X[self.adata.obs['gene'] == ptb].mean(axis=0, keepdims=True))
                aggregated_tpm=np.concatenate(aggregated_tpm, axis=0)
                aggregated_gene_std_tpm=aggregated_tpm.std(axis=0)
                ids_tpm = np.argsort(aggregated_gene_std_tpm)
                selected_var_gene_ids = ids_tpm[-int(0.8*self.adata.X.shape[1]):]
                selected_gene_ids = np.intersect1d(selected_gene_ids, selected_var_gene_ids)
                print('subset:',selected_gene_ids.shape[0], 'over', self.adata.X.shape[1])
                self.adata = self.adata[:,selected_gene_ids]
        
        self.ctrl_adata = self.adata[self.adata.obs['gene'] == 'non-targeting']
        self.ctrl_mean = self.ctrl_adata.X.mean(axis=0, keepdims=True)
        pyg_path = os.path.join(data_path, 'data_pyg')
        print('Total cell number:', len(self.adata.obs['gene']))
        self.condition_names = np.unique(self.adata.obs['gene'])
        print('Unique perturbation number:', self.condition_names.shape)
        self.set_pert_genes()
        print('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        self.not_in_go_pert = [x for x in self.condition_names if x not in self.pert_names]
        print(len(self.not_in_go_pert), self.not_in_go_pert)
        
        self.in_go_pert = [x for x in self.condition_names if x in self.pert_names]
        self.gene_names = self.adata.var.gene_name
        self.avg_sc = avg_sc
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, data_name+'_graphs.pkl')
        if subset:
            dataset_fname = dataset_fname.replace('_graphs.pkl', '_subset_graphs.pkl')
        if self.avg_sc:
            dataset_fname = dataset_fname.replace('_graphs.pkl', '_avg_sc_graphs.pkl')

        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")



    def create_train_matrix(self):
        train_Y = []
        for p in self.set2conditions['train']:
            for i in range(len(self.dataset_processed[p])):
                train_Y.append(self.dataset_processed[p][i].y)
        self.train_Y = torch.cat(train_Y, dim=0)
        

    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      validation_in_train_fraction = 0.1):

        self.seed = seed
        set_all_seeds(self.seed)
        len_perts = len(self.in_go_pert)
        rand_ids = np.random.permutation(len_perts) 
        tr_N = int(train_gene_set_size*len_perts)
        te_N = len_perts - tr_N
        va_N = int(validation_in_train_fraction*tr_N)
        tr_N = tr_N - va_N
        split_ids = {}
        split_ids['train'] = rand_ids[:tr_N]
        split_ids['val'] = rand_ids[tr_N:(tr_N+va_N)]
        split_ids['test'] = rand_ids[(tr_N+va_N):]
        self.set2conditions = {'train':[], 'test':[], 'val':[]}
        for k in split_ids.keys():
            for i in split_ids[k]:
                self.set2conditions[k].append(self.in_go_pert[i])
        print_sys("Creating training matrix")
        self.create_train_matrix()

    def tpm2OE_avg(self, matOE=None):
        if matOE is not None:
            for p in self.dataset_processed.keys():
                for i in range(len(self.dataset_processed[p])):
                    self.dataset_processed[p][i].y = self.dataset_processed[p][i].y @ matOE
                    self.dataset_processed[p][i].x = self.dataset_processed[p][i].x.T @ matOE
            self.ctrl_mean = self.ctrl_mean @ matOE

        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.ctrl_adata.var.gene_name)}

        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            print('no split is not implemented')
            exit()
        else:
            splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True) #, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
            self.dataloader =  {'train_loader': train_loader,
                                'val_loader': val_loader,
                                'test_loader': test_loader}

            print_sys("Done!")


    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'non-targeting']
        except:
            print('cannot find pert:', pert_category)
            pert_idx = None
            
        return pert_idx


    def create_cell_graph(self, X, y, pert, pert_idx=None):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X)
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category):

        adata_ = split_adata[split_adata.obs['gene'] == pert_category]
        # Create cell graphs
        cell_graphs = []
        if self.avg_sc:
            # When considering a non-control perturbation
            if pert_category != 'non-targeting':
                # Get the indices of applied perturbation
                pert_idx = self.get_pert_idx(pert_category)
                # Store list of genes that are most differentially expressed for testing
                X = self.ctrl_mean
                y = adata_.X.mean(axis=0,keepdims=True)
    
    
            # When considering a control perturbation
            else:
                pert_idx = None
                X, y = self.ctrl_mean, self.ctrl_mean        
            
            cell_graphs.append(self.create_cell_graph(np.array(X),
                                np.array(y), pert_category, pert_idx))
    
        else:
            Xs = []
            ys = []
    
            # When considering a non-control perturbation
            N = adata_.X.shape[0]
            if pert_category != 'non-targeting':
                # Get the indices of applied perturbation
                pert_idx = self.get_pert_idx(pert_category)
    
                # Store list of genes that are most differentially expressed for testing
                ctrl_samples = self.ctrl_adata[np.random.randint(0,len(self.ctrl_adata),N), :]
                for i in range(N):
                    Xs.append(ctrl_samples[[i]])
                    ys.append(adata_.X[[i]])
    
            # When considering a control perturbation
            else:
                pert_idx = None
                for cell_z in adata_.X:
                    Xs.append(cell_z)
                    ys.append(cell_z)
    
            for X, y in zip(Xs, ys):
                cell_graphs.append(self.create_cell_graph(X.toarray(),
                                    y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs


    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}         
        for p in self.condition_names:
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Done!")





class PertData:
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    data_path: str
        Path to save/load data
    gene_set_path: str
        Path to gene set to use for perturbation graph
    default_pert_graph: bool
        Whether to use default perturbation graph or not
    dataset_name: str
        Name of dataset
    dataset_path: str
        Path to dataset
    adata: AnnData
        AnnData object containing dataset
    dataset_processed: bool
        Whether dataset has been processed or not
    ctrl_adata: AnnData
        AnnData object containing control samples
    gene_names: list
        List of gene names
    node_map: dict
        Dictionary mapping gene names to indices
    split: str
        Split type
    seed: int
        Seed for splitting
    subgroup: str
        Subgroup for splitting
    train_gene_set_size: int
        Number of genes to use for training

    """
    
    def __init__(self, data_path='/oak/stanford/groups/ljerby/dzhu/Code/notebook-scGPT/data', 
                 gene_set_path=None, 
                 default_pert_graph=True):
        """
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """

        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}
        self.avg_sc = True
        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, data_name = None, data_path = None, avg_sc=True):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path

        Parameters
        ----------
        data_name: str
            Name of dataset
        data_path: str
            Path to dataset

        Returns
        -------
        None

        """
        
        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        self.avg_sc = avg_sc
        if self.avg_sc:
            dataset_fname = dataset_fname.replace('_graphs.pkl', '_avg_sc_graphs.pkl')
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        self.ctrl_mean = self.ctrl_adata.X.mean(axis=0)
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
            
    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de = False):
        """
        Process new dataset

        Parameters
        ----------
        dataset_name: str
            Name of dataset
        adata: AnnData object
            AnnData object containing gene expression data
        skip_calc_de: bool
            If True, skip differential expression calculation

        Returns
        -------
        None

        """
        
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = get_DE_genes(adata, skip_calc_de)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs_aggragated.pkl')
        print_sys("Creating pyg object for each cell in the data...")
        self.create_dataset_file()
        print_sys("Saving new dataset pyg object at " + dataset_fname) 
        pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
        print_sys("Done!")
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}

        Returns
        -------
        None

        """
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        
        if split == 'custom':
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                    raise ValueError('Please set split_dict_path for custom split')
            return
            
        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'
                 
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = False)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")

    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        de_idx: np.ndarray
            DE gene indices
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X).T
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)


    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Done!")


    def create_cell_graph_dataset(self, split_adata, pert_category):

        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        # Create cell graphs
        cell_graphs = []
        if self.avg_sc:
            # When considering a non-control perturbation
            if pert_category != 'ctrl':
                # Get the indices of applied perturbation
                pert_idx = self.get_pert_idx(pert_category)
                # Store list of genes that are most differentially expressed for testing
                X = self.ctrl_mean
                y = adata_.X.mean(axis=0)
                
    
            # When considering a control perturbation
            else:
                pert_idx = None
                X, y = self.ctrl_mean, self.ctrl_mean        
            cell_graphs.append(self.create_cell_graph(np.array(X),
                                np.array(y), None, pert_category, pert_idx))
    
        else:
            Xs = []
            ys = []
    
            # When considering a non-control perturbation
            N = adata_.X.shape[0]
            if pert_category != 'ctrl':
                # Get the indices of applied perturbation
                pert_idx = self.get_pert_idx(pert_category)
    
                # Store list of genes that are most differentially expressed for testing
                ctrl_samples = self.ctrl_adata[np.random.randint(0,len(self.ctrl_adata),N), :]
                for i in range(N):
                    Xs.append(ctrl_samples[[i]])
                    ys.append(adata_.X[[i]])
    
            # When considering a control perturbation
            else:
                pert_idx = None
                for cell_z in adata_.X:
                    Xs.append(cell_z)
                    ys.append(cell_z)
    
            for X, y in zip(Xs, ys):
                cell_graphs.append(self.create_cell_graph(X.toarray(),
                                    y.toarray(), None, pert_category, pert_idx))

        return cell_graphs
