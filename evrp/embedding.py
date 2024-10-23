import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from rl4co.models.nn.ops import PositionalEncoding
from rl4co.models.nn.env_embeddings.context import EnvContext

from rl4co.utils.ops import gather_by_index

class EVRInitEmbedding(nn.Module):
    """Initial embedding for the Electric Vehicles Routing Planning Problems (EVR).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the bus stop
        - 
    """

    def __init__(self,node_dim=2, edge_dim=2, embed_dim=128, linear_bias=True):
        super(EVRInitEmbedding, self).__init__()
        node_dim = 2 # x, y
        edge_dim = 2 # charging or not charging
        self.init_embed = CustomEdgeFeatureGNN(node_dim, edge_dim, embed_dim)

    def forward(self, td, edge_index, edge_attr):
        batch_size = td["locs"].shape[0]
        out = []
        for i in range(batch_size):
            sample_out = self.init_embed(td["locs"][i], edge_index[i], edge_attr[i])
            out.append(sample_out)
        out = torch.stack(out)
        return out

class CustomEdgeFeatureGNN(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, embed_dim):
        super(CustomEdgeFeatureGNN, self).__init__(aggr='mean')  # "mean" aggregation.
        self.lin = nn.Linear(2 * node_in_channels + edge_in_channels, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index=edge_index.to(torch.device('cuda')))

        # Start propagating messages.
        return self.propagate(edge_index=edge_index.to(torch.device('cuda:0')), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr has shape [E, edge_in_channels]

        # Combine node features and edge features
        return self.lin(torch.cat([x_i, x_j, edge_attr], dim=-1))
    
class MdppContext(EnvContext):
    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(MdppContext, self).__init__(embed_dim, step_context_dim, linear_bias)


    def forward(self, node_embeds, td):
        return self.project_context(node_embeds)
    