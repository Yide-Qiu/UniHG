
import torch
import pdb 
import pickle as pkl
import dgl
import dgl.function as fn
import time
from tqdm import tqdm
import json
from utils import entity_select


global device
device = 'cuda'

feature = pkl.load(open('feature_all/feature_all_pca.pkl', 'rb'))
edge_idx_r = pkl.load(open('relational_edge_index3.pkl', 'rb'))
edge_feat = pkl.load(open('edge_feature/edge_feature_pca.pkl', 'rb'))
print(edge_feat.shape)
edge_feat = torch.cat((edge_feat,torch.ones((1,edge_feat.shape[1]))), dim=0)

src_list = edge_idx_r[0].int()
dst_list = edge_idx_r[1].int()
link_list = edge_idx_r[2].int()
link_list = torch.cat((link_list,torch.tensor([2082], dtype=torch.int32)), dim=0)


graph = dgl.graph((src_list, dst_list))
graph.add_edges(torch.tensor([77312473], dtype=torch.int32), torch.tensor([77312473], dtype=torch.int32))

num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
edge_indices = torch.arange(num_edges)

graph.ndata['feat'] = feature
graph.edata['edge_index'] = edge_indices

edge_f = edge_feat[link_list]
graph.edata['feat'] = edge_f


def relational_message_func(edges):
    return {'msg': edges.src['feat'] * edges.data['feat']}
    
def message_func(edges):
    return {'msg': edges.src['feat']}

def reduce_func(nodes):
    return {'agg_feat': torch.mean(nodes.mailbox['msg'], dim=1)}

def relational_graph_propagation(graph):
    batch_size = 3000000
    feature_matrix = graph.ndata['feat']
    for i in tqdm(range(0, num_nodes, batch_size)):
        batch_nodes = torch.arange(i, min(i + batch_size, num_nodes)).to(torch.int32)
        batch_graph = dgl.node_subgraph(graph, batch_nodes)
        # batch_features = feature_matrix[batch_nodes] 没必要？
        # batch_graph.ndata['feat'] = batch_features
        # pdb.set_trace()
        batch_graph = batch_graph.to('cuda')
        # batch_graph.ndata['feat'] = batch_graph.ndata['feat'].cuda()
        # batch_graph.edata['edge_index'] = batch_graph.edata['edge_index'].cuda()
        batch_graph.update_all(relational_message_func, reduce_func)
        # batch_graph.send(batch_graph.edges(), relational_message_func)
        # batch_graph.recv(batch_graph.nodes(), reduce_func)
        batch_graph.ndata['new_feat'] = batch_graph.ndata['feat'] + batch_graph.ndata['agg_feat']
        feature_matrix[batch_nodes] = batch_graph.ndata['new_feat'].to('cpu')
    return feature_matrix

num_hops = 4
for h in range(1,num_hops+1):
    feature = relational_graph_propagation(graph)
    graph.ndata['feat'] = feature
    torch.save(graph.ndata['feat'], f'feature_all/feature_all_pca_{h}.pth')

pdb.set_trace()



