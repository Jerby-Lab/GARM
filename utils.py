import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from multiprocessing import Pool


def set_all_seeds(SEED):
  # REPRODUCIBILITY
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def aggregated_eval_row(pred, Y, perturbs):
    mse_dict = {}
    pearson_dict = {}
    spearman_dict = {}
    auroc_up_dict = {}
    auprc_up_dict = {}
    auroc_down_dict = {}
    auprc_down_dict = {}
    up_thres = np.quantile(Y,0.8,axis=-1)
    down_thres = np.quantile(Y,0.2,axis=-1)
    for i in range(Y.shape[0]):
        mse_dict[perturbs[i]] = torch.nn.functional.mse_loss(pred[i,:], Y[i,:]).detach().numpy()
        pearson_dict[perturbs[i]] = pearsonr(pred[i,:].detach().numpy(), Y[i,:].numpy())[0]
        spearman_dict[perturbs[i]] = spearmanr(pred[i,:].detach().numpy(), Y[i,:].numpy()).statistic
        try:
            auroc_up_dict[perturbs[i]] = roc_auc_score(Y[i,:].numpy()>up_thres[i], pred[i,:].detach().numpy())
        except:
            auroc_up_dict[perturbs[i]] = np.nan
        auprc_up_dict[perturbs[i]] = average_precision_score(Y[i,:].numpy()>up_thres[i], pred[i,:].detach().numpy())
        try:
            auroc_down_dict[perturbs[i]] = roc_auc_score(Y[i,:].numpy()<down_thres[i], -pred[i,:].detach().numpy())
        except:
            auroc_down_dict[perturbs[i]] = np.nan
        auprc_down_dict[perturbs[i]] = average_precision_score(Y[i,:].numpy()<down_thres[i], -pred[i,:].detach().numpy())
    metrics = {}
    metrics['mse_dict'] = mse_dict
    metrics['pearson_dict'] = pearson_dict
    metrics['spearman_dict'] = spearman_dict
    metrics['auroc_up_dict'] = auroc_up_dict
    metrics['auprc_up_dict'] = auprc_up_dict
    metrics['auroc_down_dict'] = auroc_down_dict
    metrics['auprc_down_dict'] = auprc_down_dict
    aggregated_metrics = {}
    aggregated_metrics['mse'] = np.mean(list(mse_dict.values()))
    aggregated_metrics['pearson'] = np.mean(list(pearson_dict.values()))
    aggregated_metrics['spearman'] = np.mean(list(spearman_dict.values()))
    aggregated_metrics['auroc_up'] = np.mean(list(auroc_up_dict.values()))
    aggregated_metrics['auprc_up'] = np.mean(list(auprc_up_dict.values()))
    aggregated_metrics['auroc_down'] = np.mean(list(auroc_down_dict.values()))
    aggregated_metrics['auprc_down'] = np.mean(list(auprc_down_dict.values()))
    return metrics, aggregated_metrics


def aggregated_eval_col(pred, Y):
    col_mse = []
    col_pearson = []
    col_spearman = []
    col_auroc_up = []
    col_auprc_up = []
    col_auroc_down = []
    col_auprc_down = []
    ids = []
    ids_spearman = []
    noise_ids = []
    up_thres = np.quantile(Y,0.8,axis=0)
    down_thres = np.quantile(Y,0.2,axis=0)
    for i in range(Y.shape[1]):
        p = pred[:,i].detach()
        if p.std() == 0:
            p += torch.rand(p.shape)*1e-10
            noise_ids.append(i)
        mse = torch.nn.functional.mse_loss(p, Y[:,i]).detach().numpy()
        auroc_up = roc_auc_score(Y[:,i].numpy()>up_thres[i], pred[:,i].detach().numpy())
        auprc_up = average_precision_score(Y[:,i].numpy()>up_thres[i], pred[:,i].detach().numpy())
        auroc_down = roc_auc_score(Y[:,i].numpy()<down_thres[i], -pred[:,i].detach().numpy())
        auprc_down = average_precision_score(Y[:,i].numpy()<down_thres[i], -pred[:,i].detach().numpy())
        pearson = pearsonr(p.numpy(), Y[:,i].numpy())[0]
        spearman = spearmanr(p.numpy(), Y[:,i].numpy()).statistic
        if ~np.isnan(pearson):
            ids.append(i)
        if ~np.isnan(spearman):
            ids_spearman.append(i)
        col_mse.append(mse)
        col_pearson.append(pearson)
        col_spearman.append(spearman)
        col_auroc_up.append(auroc_up)
        col_auprc_up.append(auprc_up)
        col_auroc_down.append(auroc_down)
        col_auprc_down.append(auprc_down)
    col_mse = np.array(col_mse)
    col_pearson = np.array(col_pearson)
    col_spearman = np.array(col_spearman)
    col_auroc_up = np.array(col_auroc_up)
    col_auprc_up = np.array(col_auprc_up)
    col_auroc_down = np.array(col_auroc_down)
    col_auprc_down = np.array(col_auprc_down)
    metrics = {}
    metrics['col_mse'] = col_mse
    metrics['col_pearson'] = col_pearson
    metrics['col_spearman'] = col_spearman
    metrics['col_auroc_up'] = col_auroc_up
    metrics['col_auprc_up'] = col_auprc_up
    metrics['col_auroc_down'] = col_auroc_down
    metrics['col_auprc_down'] = col_auprc_down
    if len(noise_ids)>0:
        print('Add noise to pred gene:', np.intersect1d(noise_ids, ids))
    print('Calculated Pearson: '+str(len(ids))+'/'+str(len(col_mse)))
    print('Calculated Spearman: '+str(len(ids_spearman))+'/'+str(len(col_mse)))
    aggregated_metrics = {}
    aggregated_metrics['mse'] = np.mean(col_mse)
    aggregated_metrics['pearson'] = np.mean(col_pearson[ids])
    aggregated_metrics['spearman'] = np.mean(col_spearman[ids_spearman])
    aggregated_metrics['auroc_up'] = np.mean(col_auroc_up)
    aggregated_metrics['auprc_up'] = np.mean(col_auprc_up)
    aggregated_metrics['auroc_down'] = np.mean(col_auroc_down)
    aggregated_metrics['auprc_down'] = np.mean(col_auprc_down)
    return metrics, aggregated_metrics




def OE2mat(OE_signatures, column_names, gene2idx):
    matOE = []
    N = max(list(gene2idx.values()))+1
    for k in column_names:
        if k in ['KO', 'mTIL']:
            continue
        signature = OE_signatures[k]
        score = np.zeros([N, 1])
        ct = 0
        for sig in signature:
            idx = gene2idx.get(sig)
            if idx is not None:
                score[idx,0] = 1
                ct += 1
        if ct > 0:
            matOE.append(score/ct)
    matOE = np.concatenate(matOE, axis=-1)
    mTIL = matOE[:,[0]] - matOE[:,[1]]
    matOE = np.concatenate([mTIL, matOE], axis=-1)
    return matOE



def pred2OE(pred, OE_signatures, column_names, gene2idx):
    pred_OE = []
    for k in column_names:
        if k in ['KO', 'mTIL']:
            continue
        signature = OE_signatures[k]
        score = np.zeros([pred.shape[0], 1])
        ct = 0
        for sig in signature:
            idx = gene2idx.get(sig)
            if idx is not None:
                score = score + pred[:,[idx]]
                ct += 1
        if ct > 0:
            pred_OE.append(score/ct)
    pred_OE = np.concatenate(pred_OE, axis=-1)
    mTIL = pred_OE[:,[0]] - pred_OE[:,[1]]
    pred_OE = np.concatenate([mTIL, pred_OE], axis=-1)
    return pred_OE


def get_gene_idx(pert_data):
  gene2idx={}
  for i in range(pert_data.gene_names.shape[0]):
      if gene2idx.get(pert_data.gene_names[i]) is not None:
          print('repeat', pert_data.gene_names[i], gene2idx.get(pert_data.gene_names[i]),'updated as', str(i))
      gene2idx[pert_data.gene_names[i]]=i
  return gene2idx

def pert2idx(gene2idx, pert, Y):
    P = []
    for p in pert:
        pt = p.split('+')
        vec = torch.zeros([Y.shape[-1]], dtype=torch.double)
        for ptb in pt:
            if ptb != 'ctrl':
                if gene2idx.get(ptb) is not None:
                    pert_idx = gene2idx[ptb]
                    vec[pert_idx] += 1
        P.append(vec)
    P = torch.stack(P).to(torch.float)
    return P

    
def get_aggregated_data(pert_data, split='train', delta=False):
    dict_ct = {}
    dict_mean = {}
    dict_var = {}
    split = split+'_loader'
    for idx, batch in enumerate(pert_data.dataloader[split]):
      x, y, pert = batch.x, batch.y, batch.pert
      if delta:
          y = y - x.view(y.shape)
      bs = len(pert) 
      for i in range(bs):
          if dict_ct.get(pert[i]) == None:
              dict_ct[pert[i]] = 1
              dict_mean[pert[i]] = y[i,:]
          else:
              dict_ct[pert[i]] += 1
              dict_mean[pert[i]] += y[i,:]
            
    for pert in dict_ct.keys():
      dict_mean[pert] = dict_mean[pert]/dict_ct[pert]
    
    for idx, batch in enumerate(pert_data.dataloader[split]):
      x, y, pert = batch.x, batch.y, batch.pert
      bs = y.shape[0]
      for i in range(bs):
          if dict_var.get(pert[i]) == None:
              dict_var[pert[i]] = (y[i,:] - dict_mean[pert[i]])**2
          else:
              dict_var[pert[i]] += (y[i,:] - dict_mean[pert[i]])**2
    
    for pert in dict_ct.keys():
      dict_var[pert] = dict_var[pert]/(dict_ct[pert]-1)
    perturb_Y = []
    perturb_V = []
    perturbs = np.sort(list(dict_ct.keys()))
    for k in perturbs:
      perturb_Y.append(dict_mean[k])
      perturb_V.append(dict_var[k])
    perturb_Y = torch.stack(perturb_Y)
    perturb_V = torch.stack(perturb_V)
    return perturb_Y, perturb_V, perturbs


############################## Helper Functions from GEARS ####################################

def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert

def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]

def combine_res(res_1, res_2):
    res_out = {}
    for key in res_1:
        res_out[key] = np.concatenate([res_1[key], res_2[key]])
    return res_out

def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]


def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

def zip_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for zip file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset
    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.zip')
        print_sys('Extracting zip file...')
        with ZipFile((save_path + '.zip'), 'r') as zip:
            zip.extractall(path = data_path)
        print_sys("Done!")  

def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print_sys('Extracting tar file...')
        with tarfile.open(save_path  + '.tar.gz') as tar:
            tar.extractall(path= data_path)
        print_sys("Done!")  

def get_go_auto(gene_list, data_path, data_name):
    """
    Get gene ontology data

    Args:
        gene_list (list): list of gene names
        data_path (str): the path to save the extracted dataset
        data_name (str): the name of the dataset

    Returns:
        df_edge_list (pd.DataFrame): gene ontology edge list
    """
    go_path = os.path.join(data_path, data_name, 'go.csv')
    
    if os.path.exists(go_path):
        return pd.read_csv(go_path)
    else:
        ## download gene2go.pkl
        if not os.path.exists(os.path.join(data_path, 'gene2go.pkl')):
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
            dataverse_download(server_path, os.path.join(data_path, 'gene2go.pkl'))
        with open(os.path.join(data_path, 'gene2go.pkl'), 'rb') as f:
            gene2go = pickle.load(f)

        gene2go = {i: list(gene2go[i]) for i in gene_list if i in gene2go}
        edge_list = []
        for g1 in tqdm(gene2go.keys()):
            for g2 in gene2go.keys():
                edge_list.append((g1, g2, len(np.intersect1d(gene2go[g1],
                   gene2go[g2]))/len(np.union1d(gene2go[g1], gene2go[g2]))))

        edge_list_filter = [i for i in edge_list if i[2] > 0]
        further_filter = [i for i in edge_list if i[2] > 0.1]
        df_edge_list = pd.DataFrame(further_filter).rename(columns = {0: 'gene1',
                                                                      1: 'gene2',
                                                                      2: 'score'})

        df_edge_list = df_edge_list.rename(columns = {'gene1': 'source',
                                                      'gene2': 'target',
                                                      'score': 'importance'})
        df_edge_list.to_csv(go_path, index = False)        
        return df_edge_list

class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list


def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(network_type, adata, threshold, k,
                           data_path, data_name, split, seed, train_gene_set_size,
                           set2conditions, default_pert_graph=True, pert_list=None, train_Y=None, gene_list=None):
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, k,
                                                     data_path, data_name, split,
                                                     seed, train_gene_set_size,
                                                     set2conditions, train_Y=train_Y, gene_list=gene_list)
    elif network_type == 'go':
        if default_pert_graph:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
            tar_data_download_wrapper(server_path, 
                                     os.path.join(data_path, 'go_essential_all'),
                                     data_path)
            df_jaccard = pd.read_csv(os.path.join(data_path, 
                                     'go_essential_all/go_essential_all.csv'))

        else:
            df_jaccard = make_GO(data_path, pert_list, data_name)

        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                    ['importance'])).reset_index(drop = True)

    return df_out


def get_coexpression_network_from_train(adata, threshold, k, data_path,
                                        data_name, split, seed, train_gene_set_size,
                                        set2conditions, train_Y=None, gene_list=None):
    """
    Infer co-expression network from training data

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
        data_path (str): path to data
        data_name (str): name of dataset
        split (str): split of dataset
        seed (int): seed for random number generator
        train_gene_set_size (int): size of training gene set
        set2conditions (dict): dictionary of perturbations to conditions
    """
    
    fpath = os.path.join(data_path, data_name)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fname = os.path.join(fpath, split + '_'  +
                     str(seed) + '_' + str(train_gene_set_size) + '_' +
                     str(threshold) + '_' + str(k) +
                     '_co_expression_network.csv')
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        if train_Y is not None and gene_list is not None:
            idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
            X_tr = train_Y.numpy()
        else:
            gene_list = [f for f in adata.var.gene_name.values]
            idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
            X = adata.X
            train_perts = set2conditions['train']
            X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
            X_tr = X_tr.toarray()
        #gene_list = adata.var['gene_name'].values

        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source',
                                                                1: 'target',
                                                                2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression


def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations
    """

    if condition == 'ctrl' or condition == 'non-targeting':
        return True
    else:
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        num_ctrl = (cond1 == 'ctrl') + (cond2 == 'ctrl')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)
    
def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create a perturbation specific cell graph for inference

    Args:
        X (np.array): gene expression matrix
        pert_idx (list): list of perturbation indices
        pert_gene (list): list of perturbations

    """

    if pert_idx is None:
        pert_idx = [-1]
    return Data(x=torch.Tensor(X).T, pert_idx = pert_idx, pert=pert_gene)
 
def create_cell_graph_dataset_for_prediction(pert_gene, ctrl_adata, gene_names,
                                             device, num_samples = 300):
    """
    Create a perturbation specific cell graph dataset for inference

    Args:
        pert_gene (list): list of perturbations
        ctrl_adata (anndata): control anndata
        gene_names (list): list of gene names
        device (torch.device): device to use
        num_samples (int): number of samples to use for inference (default: 300)

    """

    # Get the indices (and signs) of applied perturbation
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    #Xs = ctrl_adata.X.toarray().mean(axis=0, keepdims=True)
    # Create cell graphs
    cell_graphs = [create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs]
    return cell_graphs


def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))

def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

def get_DE_genes(adata, skip_calc_de):
    adata.obs.loc[:, 'dose_val'] = adata.obs.condition.apply(lambda x: '1+1' if len(x.split('+')) == 2 else '1')
    adata.obs.loc[:, 'control'] = adata.obs.condition.apply(lambda x: 0 if len(x.split('+')) == 2 else 1)
    adata.obs.loc[:, 'condition_name'] =  adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1) 
    
    adata.obs = adata.obs.astype('category')
    if not skip_calc_de:
        rank_genes_groups_by_cov(adata, 
                         groupby='condition_name', 
                         covariate='cell_type', 
                         control_group='ctrl_1', 
                         n_genes=len(adata.var),
                         key_added = 'rank_genes_groups_cov_all')
    return adata


def get_dropout_non_zero_genes(adata):
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns['rank_genes_groups_cov_all'].keys():
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups_cov_all'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
    
    return adata

class DataSplitter():
    """
    Class for handling data splitting. This class is able to generate new
    data splits and assign them as a new attribute to the data file.
    """
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_pert_genes=None,
                   test_perts=None, split_name='split', seed=None, val_size = 0.1,
                   train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75, only_test_set_perts = False):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if
                        p != 'ctrl']
        
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('ctrl')
        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self.get_simulation_split_single(unique_perts,
                                                                  train_gene_set_size,
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)
        elif self.split_type == 'no_test':
            train, val = self.get_split_list(unique_perts,
                                          test_size=val_size)      
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        if self.split_type != 'no_test':
            map_dict.update({x: 'test' for x in test})
        map_dict.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata

    def get_simulation_split_single(self, pert_list, train_gene_set_size = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)  
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        assert len(unseen_single) + len(pert_single_train) == len(pert_list)
        
        return pert_single_train, unseen_single, {'unseen_single': unseen_single}

    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)                
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)

        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}

    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')
        
        elif self.split_type == 'no_test':
            if test_perts is None:
                test_perts = np.random.choice(pert_list,
                                    int(len(pert_list) * test_size))
        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
                else:
                    test_perts = np.array(test_perts)
        else:
            if test_perts is None:
                test_perts = np.random.choice(combo_perts,
                                    int(len(combo_perts) * test_size))
        
        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        return perts

    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)