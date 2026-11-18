import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class DenseRouter(nn.Module):
    def __init__(self, input_channels, expert_num):
        super().__init__()
        self.w_gate = nn.Parameter(torch.zeros(input_channels, expert_num), requires_grad=True)
        self.b_gate = nn.Parameter(torch.zeros(expert_num), requires_grad=True)
        init.xavier_uniform_(self.w_gate)
        init.zeros_(self.b_gate)


    def forward(self, x, train=True):
        logits = x @ self.w_gate + self.b_gate
        gates = F.softmax(logits, dim=1)
        return gates



class DenseEdgeRouter(nn.Module):
    def __init__(self, input_channels, pe_dim, expert_num):
        super().__init__()
        self.w_src = nn.Parameter(torch.zeros(input_channels, pe_dim))
        self.w_dst = nn.Parameter(torch.zeros(input_channels, pe_dim))
        # Attract / Repel / Prune
        self.w_role = nn.Parameter(torch.zeros(pe_dim * 2, expert_num))
        self.b_role = nn.Parameter(torch.zeros(expert_num))
        # Attract strength / Repel strength
        self.w_strength = nn.Parameter(torch.zeros(pe_dim * 2, 2))
        self.b_strength = nn.Parameter(torch.zeros(2))

        init.xavier_uniform_(self.w_src)
        init.xavier_uniform_(self.w_dst)
        init.xavier_uniform_(self.w_role)
        init.xavier_uniform_(self.w_strength)
        init.zeros_(self.b_role)
        init.zeros_(self.b_strength)

    # def forward(self, H, H_pe, edge_index, tau=1.0):
    #     pe_dim = H_pe.shape[1]
    #     head_weight = torch.cat([self.w_role, self.w_strength], dim=1)
    #     head_bias = torch.cat([self.b_role, self.b_strength], dim=0)

    #     node_weight = head_weight[:pe_dim]
    #     pe_weight = head_weight[pe_dim:]

    #     src_output = H @ (self.w_src @ node_weight)
    #     dst_output = H @ (self.w_dst @ node_weight)

    #     edge_output = (
    #         src_output[edge_index[0]]
    #         + dst_output[edge_index[1]]
    #         + H_pe @ pe_weight
    #         + head_bias
    #     )

    #     role_logits = edge_output[:, :3]
    #     strengths = torch.sigmoid(edge_output[:, 3:])

    #     role_probs = F.softmax(role_logits, dim=1)
    #     if self.training:
    #         roles = F.gumbel_softmax(role_logits, tau=tau, hard=True, dim=1)
    #     else:
    #         route_id = role_logits.argmax(dim=1, keepdim=True)
    #         roles = torch.zeros_like(role_logits)
    #         roles.scatter_(1, route_id, 1.0)

    #     return roles, role_probs, strengths

    def forward(self, H, H_pe, edge_index, tau=1.0):
        H_src = H @ self.w_src
        H_dst = H @ self.w_dst
        H_feat = H_src[edge_index[0]] + H_dst[edge_index[1]]
        edge_feat = torch.cat([H_feat, H_pe], dim=1)

        role_logits = edge_feat @ self.w_role + self.b_role
        strengths = torch.sigmoid(
            edge_feat @ self.w_strength + self.b_strength
        )

        role_probs = F.softmax(role_logits, dim=1)
        if self.training:
            roles = F.gumbel_softmax(role_logits, tau=tau, hard=True, dim=1)
        else:
            route_id = torch.argmax(role_logits, dim=1, keepdim=True)
            roles = torch.zeros_like(role_logits)
            roles.scatter_(1, route_id, 1.0)

        return roles, role_probs, strengths
