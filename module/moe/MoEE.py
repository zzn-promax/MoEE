import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math
import time
import copy
from torch_scatter import segment_csr
from sklearn.metrics import accuracy_score as ACC
from utils import sparse_mx_to_torch_sparse_tensor, row_normalized_adjacency_

from .router import *


_GRAPH_CACHE = {}


class MoEE(nn.Module):

    def __init__(self, input_channels, class_num, device, args) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.hop_num = args.hop_num
        self.lam = args.lam
        self.pe_dim = args.pe_dim
        self.r = args.r

        self.device = device
        self.class_num = class_num
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.epochs = args.epochs
        self.patience = args.patience

        self.file_path = args.ppr_path

        #---------------- Model -------------------
        self.model = MoEModel_all(input_channels, args.nhidden, args.pe_dim, class_num, args.dropout, args)

    def fit(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        # graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        X = graph.ndata["feat"].to(self.device)

        self.prepare_graph(graph)

        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_state_dict = None

        t_start = time.time()
        for epoch in range(self.epochs):
            self.train()

            Z, C, (W_node, W_edge, P_edge, S_edge) = self.forward(X)
            loss_cls = loss_fn(C[self.train_mask], labels[self.train_mask])
            loss_prune = self.prune_budget_loss(P_edge)
            loss = loss_cls + self.lam * loss_prune

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.detach()
            loss_cls_value = loss_cls.detach()
            loss_prune_value = loss_prune.detach()
            del Z, C
            del W_node, W_edge, P_edge
            del loss, loss_cls, loss_prune


            [train_acc, valid_acc, test_acc] = self.test(X, labels, [self.train_mask, self.valid_mask, self.test_mask])

            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f'\nEpoch:{epoch}, Loss:{loss_value:.3f}, loss_cls: {loss_cls_value:.3f}, loss_prune: {loss_prune_value:.3f}')
                print(f'train acc: {train_acc:.3f} valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}')
            else:
                cnt += 1
                if cnt >= self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break

        t_finish = time.time()
        print("\n10 epoch cost: {:.4f}s\n".format((t_finish - t_start)/(epoch+1)*10))

        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch

    def prepare_graph(self, graph):
        cache_key = (
            self.file_path,
            self.pe_dim,
            str(self.device),
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        if cache_key in _GRAPH_CACHE:
            (self.edge_index, self.H_pe, self.original_degree) = _GRAPH_CACHE[cache_key]
            print('Reuse cached edge_index, H_pe and original_degree')
        else:
            adj = graph.adj(scipy_fmt='csr')
            adj_norm = row_normalized_adjacency_(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device).coalesce()
            time_0 = time.time()
            self.load_ppr(adj)
            time_1 = time.time()
            print(f'load ppr cost: {time_1 - time_0:.4f}s')
            _GRAPH_CACHE[cache_key] = (
                self.edge_index,
                self.H_pe,
                self.original_degree,
            )

    def load_ppr(self, adj):
        res = torch.load(self.file_path, map_location='cpu')
        meta = res['meta']
        node_num = adj.size(0)
        edge_num = adj._nnz()
        if meta['num_nodes'] != node_num:
            raise RuntimeError('PPR cache node count does not match the graph')
        if meta['num_edges'] != edge_num:
            raise RuntimeError('PPR cache edge count does not match the graph')

        ppr_values = res['ppr_value']
        ppr_ranks = res['ppr_rank']
        if ppr_values.numel() != edge_num:
            raise RuntimeError('Invalid number of cached PPR values')
        if ppr_ranks.numel() != edge_num:
            raise RuntimeError('Invalid number of cached PPR ranks')

        self.edge_index = adj.indices()
        edge_weight = adj.values()
        self.H_pe = torch.empty(edge_num, self.pe_dim + 2, dtype=torch.float32, device=self.device)

        self.positional_encoding(ppr_ranks, self.H_pe[:, :self.pe_dim])
        self.H_pe[:, self.pe_dim].copy_(ppr_values)
        self.H_pe[:, self.pe_dim + 1].copy_(edge_weight)

        self.original_degree = torch.bincount(self.edge_index[0], minlength=node_num).to(dtype=torch.float32).clamp(min=1)

        del res, ppr_values, ppr_ranks, edge_weight

    @torch.no_grad()
    def positional_encoding(self, pos_index, output):
        if self.pe_dim % 2 != 0:
            raise ValueError('pe_dim must be even')
        half_dim = self.pe_dim // 2
        device = output.device

        max_angle_elements = 4000000
        batch_size = max(1, max_angle_elements // half_dim)
        scale = math.sqrt(2.0 / self.pe_dim)
        div_term = torch.exp(torch.arange(0, self.pe_dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / self.pe_dim))

        for begin in range(0, pos_index.numel(), batch_size):
            end = min(begin + batch_size, pos_index.numel())
            position = pos_index[begin:end].to(device=device, dtype=torch.float32).unsqueeze(1)
            angles = position * div_term.unsqueeze(0)
            output[begin:end, 0::2].copy_(torch.sin(angles).mul_(scale))
            output[begin:end, 1::2].copy_(torch.cos(angles).mul_(scale))

        return output

    def prune_budget_loss(self, role_probs_list):
        prune_ratios = torch.stack([
            probs[:, 2].mean()
            for probs in role_probs_list
        ])
        return F.relu(prune_ratios - self.r).mean()

    def forward(self, X):
        Z, C, outputs = self.model(
            X,
            self.H_pe,
            self.edge_index,
            self.original_degree,
        )
        return Z, C, outputs

    def test(self, X, labels, index_list):
        self.eval()
        with torch.no_grad():
            Z, C, _ = self.forward(X)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph):
        self.eval()
        # graph = graph.to(self.device)
        X = graph.ndata['feat'].to(self.device)
        with torch.no_grad():
            Z, logits, (W_node, W_edge, P_edge, S_edge) = self.forward(X)
            y_pred = torch.argmax(logits, dim=1)
            C = F.softmax(logits, dim=1)

        print("W_node:")
        for W in W_node:
            print(torch.mean(W, dim=0))
        print("W_edge:")
        for W in W_edge:
            print(torch.mean(W, dim=0))
        print("P_edge:")
        for P in P_edge:
            print(torch.mean(P, dim=0))
        print("S_edge:")
        for S in S_edge:
            print(torch.mean(S, dim=0))

        return y_pred.cpu(), C.cpu(), Z.cpu(), (W_node, W_edge, P_edge, S_edge)



class MoEModel_all(nn.Module):

    def __init__(self, input_channels, hidden_channels, pe_dim, class_num, dropout, args):
        super().__init__()
        #------------- Parameters ----------------
        self.class_num = class_num
        self.dropout = dropout

        #---------------- Model -------------------
        self.node_encoder = nn.Linear(input_channels, hidden_channels)
        self.moe_model = MoEEModel(hidden_channels, pe_dim, args.hop_num, args.router_tau)
        self.classifier = nn.Linear(
            hidden_channels * (args.hop_num + 1),
            class_num,
        )
        self.init_params()

    def init_params(self):
        init.xavier_uniform_(self.node_encoder.weight)
        init.zeros_(self.node_encoder.bias)

    def forward(self, X, H_pe, edge_index, original_degree):
        X = F.dropout(X, p=self.dropout, training=self.training)
        H_node = self.node_encoder(X)

        H, (W_node, W_edge, P_edge, S_edge) = self.moe_model(H_node, H_pe, edge_index, original_degree)
        C = self.classifier(H)
        return H, C, (W_node, W_edge, P_edge, S_edge)



class MoEEModel(nn.Module):
    def __init__(self, channels_num, pe_dim, hop_num, router_tau):
        super().__init__()
        self.hop_num = hop_num
        self.router_tau = router_tau
        self.expert_weights_list = nn.ParameterList([
            nn.Parameter(torch.randn(3, channels_num, channels_num)) for _ in range(hop_num)
        ])
        self.expert_bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(3, channels_num)) for _ in range(hop_num)
        ])
        self.edge_router_list = nn.ModuleList([
            DenseEdgeRouter(channels_num, pe_dim + 2, 3, ) for _ in range(hop_num)
        ])
        self.node_gate_list = nn.ModuleList([
            DenseRouter(channels_num * 4, 4) for _ in range(hop_num)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(channels_num) for _ in range(hop_num)
        ])
        self.init_params()

    def init_params(self):
        for weights in self.expert_weights_list:
            for weight in weights:
                init.xavier_uniform_(weight)

        for bias in self.expert_bias_list:
            init.zeros_(bias)

    def forward(self, H_node, H_pe, edge_index, original_degree):
        num_nodes = H_node.shape[0]

        H_out_list = [H_node]
        W_edge_list = []
        W_node_list = []
        P_edge_list = []
        S_edge_list = []

        for hop in range(self.hop_num):
            roles, role_probs, strengths = self.edge_router_list[hop](
                H_node.detach(), H_pe, edge_index, tau=self.router_tau,
            )

            route_id = roles.detach().argmax(dim=1)
            attract_mask = route_id.eq(0)
            repel_mask = route_id.eq(1)

            H_trans = torch.einsum(
                'eio,bi->beo',
                self.expert_weights_list[hop],
                H_node,
            )
            H_trans = H_trans + self.expert_bias_list[hop]
            H_trans = F.relu(H_trans)
            H_ego = H_trans[:, 0, :]

            attract_weight = roles[attract_mask, 0] * strengths[attract_mask, 0]
            repel_weight = roles[repel_mask, 1] * strengths[repel_mask, 1]

            H_attract = self.expert_forward(
                H_trans[:, 1, :],
                edge_index[:, attract_mask],
                attract_weight,
                original_degree,
                repel=False,
            )
            H_repel = self.expert_forward(
                H_trans[:, 2, :],
                edge_index[:, repel_mask],
                repel_weight,
                original_degree,
                repel=True,
            )

            H_expout = [H_node, H_ego, H_attract, H_repel]
            H_expout_cat = torch.cat(H_expout, dim=1)
            node_weights = self.node_gate_list[hop](H_expout_cat)

            stacked = torch.stack(H_expout, dim=0)
            H_node = torch.einsum(
                'be,ebd->bd',
                node_weights,
                stacked,
            )
            H_node = self.layer_norms[hop](H_node)

            H_out_list.append(H_node)
            W_edge_list.append(roles)
            S_edge_list.append(strengths)
            P_edge_list.append(role_probs)
            W_node_list.append(node_weights)

        H_out = torch.cat(H_out_list, dim=1)
        return H_out, (W_node_list, W_edge_list, P_edge_list, S_edge_list)

    def expert_forward(self, H_node, edge_index, edge_weight, original_degree, repel=False):
        if edge_index.shape[1] == 0:
            return torch.zeros_like(H_node)
        src, dst = edge_index
        num_nodes = H_node.shape[0]

        active_degree = torch.bincount(src, minlength=num_nodes)
        indptr = coo_src_to_csr_indices(src, num_nodes, active_degree)
        neighbor_sum = segment_csr(
            H_node[dst] * edge_weight.unsqueeze(1),
            indptr,
            reduce='sum',
        )

        if repel:
            weight_sum = segment_csr(
                edge_weight.unsqueeze(1),
                indptr,
                reduce='sum',
            )
            output = weight_sum * H_node - neighbor_sum
        else:
            output = neighbor_sum

        return output / original_degree.unsqueeze(1)

def coo_src_to_csr_indices(src, num_nodes, degrees):
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=src.device)
    torch.cumsum(degrees, dim=0, out=indptr[1:])

    return indptr
