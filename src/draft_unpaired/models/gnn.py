"""TF-TG graph neural network for unpaired trajectories."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv


class PairNorm(nn.Module):
    """Normalize centered node features by their mean row norm."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        norm = centered.pow(2).sum(dim=1, keepdim=True).sqrt().mean()
        return self.scale * centered / (norm + 1e-6)


class TreeUnpairedTFTGModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 2,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        gat_out = hidden_channels * heads

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = HeteroConv(
            {
                ("TF", "regulate", "TG"): GATConv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("TG", "regulated_by", "TF"): GATConv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("TF", "regulate", "TG"): GATConv(
                    gat_out,
                    hidden_channels,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("TG", "regulated_by", "TF"): GATConv(
                    gat_out,
                    hidden_channels,
                    heads=heads,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )

        self.bn_tf1 = nn.BatchNorm1d(gat_out)
        self.bn_tg1 = nn.BatchNorm1d(gat_out)
        self.bn_tf2 = nn.BatchNorm1d(gat_out)
        self.bn_tg2 = nn.BatchNorm1d(gat_out)

        self.pn_tf = PairNorm(scale=1.0)
        self.pn_tg = PairNorm(scale=1.0)

        self.res_tf = nn.Linear(in_channels, gat_out)
        self.res_tg = nn.Linear(in_channels, gat_out)

        self.gru_tf = nn.GRUCell(gat_out, hidden_channels)
        self.gru_tg = nn.GRUCell(gat_out, hidden_channels)
        self.gru_skip_tf = nn.Linear(gat_out, hidden_channels)
        self.gru_skip_tg = nn.Linear(gat_out, hidden_channels)

        self.fc_tf = nn.Linear(hidden_channels, out_channels)
        self.fc_tg = nn.Linear(hidden_channels, out_channels)

    def spatial(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        x_tf0 = data["TF"].x
        x_tg0 = data["TG"].x
        x_dict = {"TF": x_tf0, "TG": x_tg0}
        edge_index_dict = {
            ("TF", "regulate", "TG"): data["TF", "regulate", "TG"].edge_index,
            ("TG", "regulated_by", "TF"): data["TG", "regulated_by", "TF"].edge_index,
        }

        h1 = self.conv1(x_dict, edge_index_dict)
        h_tf = self.dropout(self.bn_tf1(h1["TF"]))
        h_tg = self.dropout(self.bn_tg1(h1["TG"]))

        h2 = self.conv2({"TF": h_tf, "TG": h_tg}, edge_index_dict)
        h_tf2 = self.bn_tf2(h2["TF"])
        h_tg2 = self.bn_tg2(h2["TG"])

        h_tf2 = self.pn_tf(h_tf2)
        h_tg2 = self.pn_tg(h_tg2)

        h_tf2 = self.dropout(h_tf2 + self.res_tf(x_tf0))
        h_tg2 = self.dropout(h_tg2 + self.res_tg(x_tg0))
        return h_tf2, h_tg2

    def forward(
        self,
        graphs: list[Any],
        topo_order: list[int],
        parents_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_tf: dict[int, torch.Tensor] = {}
        feat_tg: dict[int, torch.Tensor] = {}
        for time_idx, graph in enumerate(graphs):
            feat_tf[time_idx], feat_tg[time_idx] = self.spatial(graph)

        hidden_tf: dict[int, torch.Tensor] = {}
        hidden_tg: dict[int, torch.Tensor] = {}
        hidden_size = self.gru_tf.hidden_size

        for time_idx in topo_order:
            x_tf = feat_tf[time_idx]
            x_tg = feat_tg[time_idx]
            parent_indices = [parent for parent in parents_list[time_idx] if parent is not None and parent >= 0]

            if len(parent_indices) == 0:
                h0_tf = x_tf.new_zeros(x_tf.size(0), hidden_size)
                h0_tg = x_tg.new_zeros(x_tg.size(0), hidden_size)
            elif len(parent_indices) == 1:
                h0_tf = hidden_tf[parent_indices[0]]
                h0_tg = hidden_tg[parent_indices[0]]
            else:
                h0_tf = torch.stack([hidden_tf[parent] for parent in parent_indices], dim=0).mean(dim=0)
                h0_tg = torch.stack([hidden_tg[parent] for parent in parent_indices], dim=0).mean(dim=0)

            hidden_tf[time_idx] = self.gru_tf(x_tf, h0_tf) + self.gru_skip_tf(x_tf)
            hidden_tg[time_idx] = self.gru_tg(x_tg, h0_tg) + self.gru_skip_tg(x_tg)

        tf_stack = torch.stack([hidden_tf[time_idx] for time_idx in topo_order], dim=1)
        tg_stack = torch.stack([hidden_tg[time_idx] for time_idx in topo_order], dim=1)
        return self.fc_tf(tf_stack), self.fc_tg(tg_stack)


def cosine_abs_prob(out_tf: torch.Tensor, out_tg: torch.Tensor, eps: float = 1e-6) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    sims: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    time_count = int(out_tf.size(1))
    for time_idx in range(time_count):
        tf_vec = F.normalize(out_tf[:, time_idx, :], p=2, dim=-1, eps=1e-8)
        tg_vec = F.normalize(out_tg[:, time_idx, :], p=2, dim=-1, eps=1e-8)
        cos = torch.matmul(tf_vec, tg_vec.t())
        prob = cos.abs().clamp(min=eps, max=1.0 - eps)
        sims.append(prob.detach().cpu())
        probs.append(prob)
    return sims, probs


def build_model(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    heads: int = 2,
    dropout_rate: float = 0.4,
) -> TreeUnpairedTFTGModel:
    return TreeUnpairedTFTGModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout_rate=dropout_rate,
    )
