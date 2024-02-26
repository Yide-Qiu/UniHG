
import torch
import pdb 
import pickle as pkl
import dgl
import dgl.function as fn
import time
from tqdm import tqdm

# Homogeneous propagation 

# full
# feature = pkl.load(open('./../feature_all_pca.pkl', 'rb'))
# edge_idx_r = torch.load('./../relational_edge_index.pth')

# 10M
# feature = torch.load('./../new_feature_10M_pca.pth')
# edge_idx_r = torch.load('./../edge_index_10M.pth')

# full
feature = torch.load('./../new_feature_1M_pca.pth')
edge_idx_r = torch.load('./../edge_index_1M.pth')

src_list = edge_idx_r[0].int()
dst_list = edge_idx_r[1].int()

graph = dgl.graph((src_list, dst_list))
# graph.add_edges(torch.tensor([77312473], dtype=torch.int32), torch.tensor([77312473], dtype=torch.int32))
# graph.add_edges(torch.tensor([10024902], dtype=torch.int32), torch.tensor([10024902], dtype=torch.int32))
# graph.add_edges(torch.tensor([10024903], dtype=torch.int32), torch.tensor([10024903], dtype=torch.int32))

num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
edge_indices = torch.arange(num_edges)

graph.ndata['feat'] = feature

def relational_message_func(edges):
    return {'msg': edges.src['feat'] * edges.data['feat']}
    
def message_func(edges):
    return {'msg': edges.src['feat']}

def reduce_func(nodes):
    return {'agg_feat': torch.mean(nodes.mailbox['msg'], dim=1)}

def relational_graph_propagation(graph):
    batch_size = 200000
    feature_matrix = graph.ndata['feat']
    for i in tqdm(range(0, num_nodes, batch_size)):
        batch_nodes = torch.arange(i, min(i + batch_size, num_nodes)).to(torch.int32)
        batch_graph = dgl.node_subgraph(graph, batch_nodes)
        batch_graph = batch_graph.to('cuda')
        batch_graph.update_all(message_func, reduce_func)
        batch_graph.ndata['new_feat'] = batch_graph.ndata['feat'] + batch_graph.ndata['agg_feat']
        feature_matrix[batch_nodes] = batch_graph.ndata['new_feat'].to('cpu')
    return feature_matrix

num_hops = 4
for h in range(1,num_hops+1):
    feature = relational_graph_propagation(graph)
    graph.ndata['feat'] = feature
    torch.save(graph.ndata['feat'], f'./../HM_feature_1M_pca_{h}.pth')

pdb.set_trace()
