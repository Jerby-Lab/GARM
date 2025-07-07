import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import SGConv


class FFNN(torch.nn.Module):
    r"""
        An implementation of Multilayer Perceptron (MLP).
    """
    def __init__(self, input_dim=1025, hidden_sizes=(256,), activation='elu', num_classes=64):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        if sum(self.hidden_sizes) > 0: # multi-layer model
            layers = []
            for i in range(len(hidden_sizes)):
                layers.append(torch.nn.Linear(input_dim, hidden_sizes[i])) 
                if activation=='relu':
                  layers.append(torch.nn.ReLU())
                elif activation=='elu':
                  layers.append(torch.nn.ELU())
                elif activation=='tanh':
                  layers.append(torch.nn.Tanh())
                else:
                  pass 
                input_dim = hidden_sizes[i]
            self.layers = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """forward pass"""
        if sum(self.hidden_sizes) > 0:
            x = self.layers(x)
        return self.fc(x), x 


class Perturb_NN_GARM_GO(torch.nn.Module):
    def __init__(self, D, hidden_sizes=(), D_pert=3072, K=256, activation='relu'):
        super().__init__()
        self.G = torch.nn.Parameter(torch.randn(K, D))
        self.K, self.D = K, D
        self.MLP_P = FFNN(input_dim=256, hidden_sizes=hidden_sizes, num_classes=K*5, activation=activation)

    def forward(self, P1, P2):
        """forward pass"""
        G = self.G
        P = P2
        B = P.shape[0]
        P,_ = self.MLP_P(P)
        P = torch.permute(P.view(B,5,self.K), (1,0,2))
        preds = []
        for i in range(5):
            Gi = G
            Pi = torch.nn.functional.tanh(P[i,:,:])
            predi = Pi @ Gi
            preds.append(predi)
        return preds

    def predict(self, pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col):
        eps = 1e-5
        mean_est_row, mean_est_col = pred_mean.mean(dim=1, keepdim=True), pred_mean.mean(dim=0, keepdim=True)
        std_est_row, std_est_col = torch.clip(pred_var_row.std(dim=1, keepdim=True), min=eps), torch.clip(pred_var_col.std(dim=0, keepdim=True), min=eps)
        pear_mean_est_row, pear_std_est_row = pred_pear_row.mean(dim=1, keepdim=True), torch.clip(pred_pear_row.std(dim=1, keepdim=True), min=eps)
        pear_mean_est_col, pear_std_est_col = pred_pear_col.mean(dim=0, keepdim=True), torch.clip(pred_pear_col.std(dim=0, keepdim=True), min=eps)
        pred_row = (pred_pear_row - pear_mean_est_row)/pear_std_est_row
        pred_row = pred_row * std_est_row + mean_est_row
        pred_col = (pred_pear_col - pear_mean_est_col)/pear_std_est_col
        pred_col = pred_col * std_est_col + mean_est_col
        pred_full = pred_row*0.5 + pred_col*0.5
        return pred_full, pred_row, pred_col



class Perturb_NN_GARM(torch.nn.Module):
    def __init__(self, D, hidden_sizes=(), D_pert=3072, K=256, activation='relu'):
        super().__init__()
        self.G = torch.nn.Parameter(torch.randn(K, D))
        self.K, self.D = K, D
        self.MLP_F = FFNN(input_dim=D_pert, hidden_sizes=hidden_sizes, num_classes=256*3, activation=activation)
        self.MLP_P = FFNN(input_dim=256*4, hidden_sizes=hidden_sizes, num_classes=K*5, activation=activation)

    def forward(self, P1, P2):
        """forward pass"""
        G = self.G
        P1,_ = self.MLP_F(P1)
        P = torch.cat([P1, P2], dim=-1)
        B = P.shape[0]
        P,_ = self.MLP_P(P)
        P = torch.permute(P.view(B,5,self.K), (1,0,2))
        preds = []
        for i in range(5):
            Gi = G
            Pi = torch.nn.functional.tanh(P[i,:,:])
            predi = Pi @ Gi
            preds.append(predi)
        return preds

    def predict(self, pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col):
        eps = 1e-5
        mean_est_row, mean_est_col = pred_mean.mean(dim=1, keepdim=True), pred_mean.mean(dim=0, keepdim=True)
        std_est_row, std_est_col = torch.clip(pred_var_row.std(dim=1, keepdim=True), min=eps), torch.clip(pred_var_col.std(dim=0, keepdim=True), min=eps)
        pear_mean_est_row, pear_std_est_row = pred_pear_row.mean(dim=1, keepdim=True), torch.clip(pred_pear_row.std(dim=1, keepdim=True), min=eps)
        pear_mean_est_col, pear_std_est_col = pred_pear_col.mean(dim=0, keepdim=True), torch.clip(pred_pear_col.std(dim=0, keepdim=True), min=eps)
        pred_row = (pred_pear_row - pear_mean_est_row)/pear_std_est_row
        pred_row = pred_row * std_est_row + mean_est_row
        pred_col = (pred_pear_col - pear_mean_est_col)/pear_std_est_col
        pred_col = pred_col * std_est_col + mean_est_col
        pred_full = pred_row*0.5 + pred_col*0.5
        return pred_full, pred_row, pred_col



class Perturb_NN_GARM_tmp(torch.nn.Module):
    def __init__(self, D, hidden_sizes=(), D_pert=3072, K=256, activation='relu'):
        super().__init__()
        self.G = torch.nn.Parameter(torch.randn(K, D))
        self.K, self.D = K, D
        self.MLP_F = FFNN(input_dim=D_pert, hidden_sizes=hidden_sizes, num_classes=256*3, activation=activation)
        self.MLP_P = FFNN(input_dim=256*4, hidden_sizes=hidden_sizes, num_classes=K, activation=activation)

    def forward(self, P1, P2):
        """forward pass"""
        G = self.G
        P1,_ = self.MLP_F(P1)
        P = torch.cat([P1, P2], dim=-1)
        P,_ = self.MLP_P(P)
        preds = P @ G
        return preds




class Perturb_NN_GARM_col(torch.nn.Module):
    def __init__(self, D, b, hidden_sizes=(), D_pert=3072, K=256, activation='relu'):
        super().__init__()
        self.G = torch.nn.Parameter(torch.randn(K, D))
        self.K, self.D = K, D
        self.b = b 
        self.MLP_F = FFNN(input_dim=D_pert, hidden_sizes=hidden_sizes, num_classes=256*3, activation=activation)
        self.MLP_P = FFNN(input_dim=256*4, hidden_sizes=hidden_sizes, num_classes=K*5, activation=activation)
        self.MLP_G = FFNN(input_dim=K, hidden_sizes=hidden_sizes, num_classes=K*5, activation=activation)

    def forward(self, P1, P2):
        """forward pass"""
        G = self.G
        P1,_ = self.MLP_F(P1)
        P = torch.cat([P1, P2], dim=-1)
        B = P.shape[0]
        P,_ = self.MLP_P(P)
        P = torch.permute(P.view(B,5,self.K), (1,0,2))
        preds = []
        for i in range(3):
            Gi = torch.nn.functional.tanh(G)
            Pi = torch.nn.functional.tanh(P[i,:,:])
            predi = Pi @ Gi + self.b
            preds.append(predi)
        return preds

    def predict(self, pred_mean, pred_var_col, pred_pear_col):
        eps = 1e-5
        mean_est_col = pred_mean.mean(dim=0, keepdim=True)
        std_est_col = torch.clip(pred_var_col.std(dim=0, keepdim=True), min=eps)
        pear_mean_est_col, pear_std_est_col = pred_pear_col.mean(dim=0, keepdim=True), torch.clip(pred_pear_col.std(dim=0, keepdim=True), min=eps)
        pred_col = (pred_pear_col - pear_mean_est_col)/pear_std_est_col
        pred_col = pred_col * std_est_col + mean_est_col
        return pred_col


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class GEARS_Model_GAR_M(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model_GAR_M, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        self.D = self.num_genes
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.D,
                                               hidden_size))
        self.indv_b1 = nn.Parameter(torch.rand(self.D))
        self.act = nn.ReLU()

        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.D, hidden_size,
                                     hidden_size])
        self.fc1 = MLP([2*hidden_size, hidden_size, 1])
        self.fc2 = MLP([2*hidden_size, hidden_size, 1])
        self.fc3 = MLP([2*hidden_size, hidden_size, 1])
        self.fc4 = MLP([2*hidden_size, hidden_size, 1])
        self.fc5 = MLP([2*hidden_size, hidden_size, 1])
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        #self.combine_w = nn.Parameter(torch.rand(2,self.D))
        #self.combine_w = nn.Parameter(torch.rand(2))
        
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())
            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb) 

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)


            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]


            base_emb = base_emb.reshape(num_graphs * self.D, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb) 
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.D, -1)
            tmp_out = out
            out = out * self.indv_w1
            w = torch.sum(out, dim = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out)
            cross_gene_embed = cross_gene_embed.repeat(1, self.D)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.D, -1])
            cross_gene_out = torch.cat([tmp_out, cross_gene_embed], 2).reshape([num_graphs * self.D, -1])

            
            
            out_1 = self.fc1(cross_gene_out)
            out_1 = out_1 + x.reshape(-1,1).to(out_1.device)
            out_1 = torch.stack(torch.split(torch.flatten(out_1), self.D))

            
            out_2 = self.fc2(cross_gene_out)
            out_2 = out_2 + x.reshape(-1,1).to(out_2.device)
            out_2 = torch.stack(torch.split(torch.flatten(out_2), self.D))

            
            out_3 = self.fc3(cross_gene_out)
            out_3 = out_3 + x.reshape(-1,1).to(out_3.device)
            out_3 = torch.stack(torch.split(torch.flatten(out_3), self.D))

            
            out_4 = self.fc4(cross_gene_out)
            out_4 = out_4 + x.reshape(-1,1).to(out_4.device)
            out_4 = torch.stack(torch.split(torch.flatten(out_4), self.D))
            
            
            out_5 = self.fc5(cross_gene_out)
            out_5 = out_5 + x.reshape(-1,1).to(out_5.device)
            out_5 = torch.stack(torch.split(torch.flatten(out_5), self.D))
            
            
            return out_1, out_2, out_3, out_4, out_5



    def predict(self, pred_mean, pred_var_row, pred_pear_row, pred_var_col, pred_pear_col):
        eps = 1e-5
        mean_est_row, mean_est_col = pred_mean.mean(dim=1, keepdim=True), pred_mean.mean(dim=0, keepdim=True)
        std_est_row, std_est_col = torch.clip(pred_var_row.std(dim=1, keepdim=True), min=eps), torch.clip(pred_var_col.std(dim=0, keepdim=True), min=eps)
        pear_mean_est_row, pear_std_est_row = pred_pear_row.mean(dim=1, keepdim=True), torch.clip(pred_pear_row.std(dim=1, keepdim=True), min=eps)
        pear_mean_est_col, pear_std_est_col = pred_pear_col.mean(dim=0, keepdim=True), torch.clip(pred_pear_col.std(dim=0, keepdim=True), min=eps)
        pred_row = (pred_pear_row - pear_mean_est_row)/pear_std_est_row
        pred_row = pred_row * std_est_row + mean_est_row
        pred_col = (pred_pear_col - pear_mean_est_col)/pear_std_est_col
        pred_col = pred_col * std_est_col + mean_est_col
        #weights = torch.exp(self.combine_w)
        #weights = weights/torch.sum(weights)
        #pred_full = pred_row*weights[0] + pred_col*weights[1]
        pred_full = pred_row*0.5 + pred_col*0.5
        #return (pred_row + pred_col)/2# pred_col
        return pred_full, pred_row, pred_col





class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        print('utlizing orginal GEARS backbone')
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1).to(out.device)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
        

