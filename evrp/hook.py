from tensordict.tensordict import TensorDict
from typing import Union
import torch

def pre_embedding_hook(td: Union[TensorDict, None]):
    
    num_nodes = td["locs"].shape[-2]
    # 生成所有可能的点对组合
    all_edges = torch.combinations(torch.arange(num_nodes), r=2)

    # 生成双向边（无向图）
    edge_index = torch.cat([all_edges, all_edges.flip(dims=[1])], dim=0).t()


    dist_features = td["dist_mat"][:,edge_index[0], edge_index[1]]
    charge_features = td["charge_mat"][:,edge_index[0], edge_index[1]]


    # 获取批次大小
    batch_size = td["locs"].shape[0]
    # 对edge_index进行拓展复制
    edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 拼接边特征
    edge_attr = torch.stack([dist_features, charge_features], dim=2)

    return edge_index, edge_attr