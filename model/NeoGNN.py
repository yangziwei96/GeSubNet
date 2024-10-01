from torch_geometric.nn import GCNConv
import torch
from torch import nn
import torch_sparse
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

class NeoGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3,
                 dropout=0.0, f_edge_dim=8, f_node_dim=128, g_phi_dim=128):
        super(NeoGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        cached = False
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))

        self.f_edge = torch.nn.Sequential(torch.nn.Linear(1, f_edge_dim).double(),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(f_edge_dim, 1).double())

        self.f_node = torch.nn.Sequential(torch.nn.Linear(1, f_node_dim).double(),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(f_node_dim, 1).double())

        self.g_phi = torch.nn.Sequential(torch.nn.Linear(1, g_phi_dim).double(),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(g_phi_dim, 1).double())

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.constant_(self.alpha, 0)
        self.f_edge.apply(self.weight_reset)
        self.f_node.apply(self.weight_reset)
        self.g_phi.apply(self.weight_reset)

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def decode(self, z, edge, A, node_struct_feat=None):
        batch_size = edge.shape[-1]
        num_nodes = z.shape[0]
        # 1. compute similarity scores of node pairs via conventionl GNNs (feature + adjacency matrix)
        #out_feat = self.predictor(z[edge[0]], z[edge[1]])
        out_feat = (z[edge[0]] * z[edge[1]]).sum(dim=-1)  # product of a pair of nodes on each edge
        # 2. compute similarity scores of node pairs via Neo-GNNs
        # 2-1. Structural feature generation
        if node_struct_feat is None:
            row_A, col_A = A.nonzero()
            tmp_A = torch.stack([torch.from_numpy(row_A), torch.from_numpy(col_A)]).type(torch.LongTensor).to(edge.device)
            row_A, col_A = tmp_A[0], tmp_A[1]
            edge_weight_A = torch.from_numpy(A.data).to(edge.device)
            edge_weight_A = self.f_edge(edge_weight_A.unsqueeze(-1))
            node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=num_nodes)

        indexes_src = edge[0].cpu().numpy()
        row_src, col_src = A[indexes_src].nonzero()
        edge_index_src = torch.stack([torch.from_numpy(row_src), torch.from_numpy(col_src)]).type(torch.LongTensor).to(edge.device)
        edge_weight_src = torch.from_numpy(A[indexes_src].data).to(edge.device)
        edge_weight_src = edge_weight_src * self.f_node(node_struct_feat[col_src]).squeeze()

        indexes_dst = edge[1].cpu().numpy()
        row_dst, col_dst = A[indexes_dst].nonzero()
        edge_index_dst = torch.stack([torch.from_numpy(row_dst), torch.from_numpy(col_dst)]).type(torch.LongTensor).to(edge.device)
        edge_weight_dst = torch.from_numpy(A[indexes_dst].data).to(edge.device)
        edge_weight_dst = edge_weight_dst * self.f_node(node_struct_feat[col_dst]).squeeze()
        
        
        mat_src = SparseTensor.from_edge_index(edge_index_src, edge_weight_src, [batch_size, num_nodes])
        mat_dst = SparseTensor.from_edge_index(edge_index_dst, edge_weight_dst, [batch_size, num_nodes])
        out_struct = (mat_src @ mat_dst.to_dense().t()).diag()
        
        out_struct = self.g_phi(out_struct.unsqueeze(-1))
        out_struct = torch.sigmoid(out_struct)

        alpha = torch.softmax(self.alpha, dim=0)
        out = alpha[0] * out_struct[0] + alpha[1] * out_feat + 1e-15

        del edge_weight_src, edge_weight_dst, node_struct_feat
        torch.cuda.empty_cache()

        return out
    
    def decode_all(self, z, A):
        num_nodes = 571
        edge_index = torch.combinations(torch.arange(num_nodes), 2).t().to(z.device)
        out = self.decode(z, edge_index, A)
        return (out > 0).nonzero(as_tuple=False).t()

    def encode(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x