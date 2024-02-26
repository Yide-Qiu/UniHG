import random
import torch
import pdb 
import pickle as pkl
import dgl
import numpy as np
from utils import entity_select

# global device
# device = 'cuda'

random.seed(123)

features = pkl.load(open('feature_all/feature_all_pca.pkl', 'rb'))
labels = torch.load('multilabels/labels_cluster.pth')
edge_idx_r = pkl.load(open('relational_edge_index3.pkl', 'rb'))

src_list = edge_idx_r[0].int()
dst_list = edge_idx_r[1].int()
graph = dgl.graph((src_list, dst_list))
graph.add_edges(torch.tensor([77312473], dtype=torch.int32), torch.tensor([77312473], dtype=torch.int32))
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
print(f'wiki-full has {num_nodes} nodes and {num_edges} edges.')
graph.ndata['feat'] = features
# full graph

pdb.set_trace()

# idx = list(range(features.shape[0]))
# random.shuffle(idx)
# num = 1000000
####################################################
node_idx = random.sample(range(num_nodes), 1000000) # no random sample node because the small edges. select top-k degree nodes.
sample_num = 1000000
in_degrees = graph.in_degrees().numpy()
topk_indices = np.argsort(in_degrees)[-int(sample_num/10):]
fine = False
base_length = 50
while fine == False:
    base_length += 10
    walks, _= dgl.sampling.random_walk(graph, nodes=topk_indices, length=base_length, restart_prob=0.0)
    tensor_set = set(walks.view(-1).tolist())
    print(f'now the length is {base_length}')
    if len(tensor_set) >= sample_num:
        fine = True
        node_idx = list(tensor_set)
sub_graph = dgl.node_subgraph(graph, node_idx)
print(f'wiki-1M has {sub_graph.number_of_nodes()} nodes and {sub_graph.number_of_edges()} edges.')
sub_features = sub_graph.ndata['feat']
sub_labels = labels[node_idx] # 
print(sub_labels)
torch.save(sub_features, 'feature_1M/feature_1M_pca.pth')
torch.save(sub_labels, 'multilabels/labels_cluster_1M.pth')

num_hops = 5
for i in range(1, num_hops):
    features = torch.load(f'feature_all/feature_all_pca_{i}.pth')
    sub_features = features[node_idx]
    torch.save(sub_features, f'feature_1M/feature_1M_pca_{i}.pth')

print('wiki-1M has been done.')
pdb.set_trace()

####################################################
# sample_num = 10000000
# in_degrees = graph.in_degrees().numpy()
# topk_indices = np.argsort(in_degrees)[-int(sample_num/5):]
# random_indices = random.sample(range(num_nodes), int(sample_num/2))
# indices = list(topk_indices) + random_indices
# fine = False
# base_length = 90
# while fine == False:
#     walks, _= dgl.sampling.random_walk(graph, nodes=indices, length=base_length, restart_prob=0.0)
#     tensor_set = set(walks.view(-1).tolist())
#     print(f'now the length is {base_length} nodes : {len(tensor_set)}')
#     if len(tensor_set) >= sample_num:
#         fine = True
#         node_idx = list(tensor_set)
#     base_length += 10
# sub_graph = dgl.node_subgraph(graph, node_idx)
# print(f'wiki-10M has {sub_graph.number_of_nodes()} nodes and {sub_graph.number_of_edges()} edges.')
# sub_features = sub_graph.ndata['feat']
# sub_labels = labels[node_idx] 
# print(sub_labels)
# torch.save(sub_features, 'feature_10M/feature_10M_pca.pth')
# torch.save(sub_labels, 'multilabels/labels_cluster_10M.pth')

# num_hops = 5
# for i in range(1, num_hops):
#     features = torch.load(f'feature_all/feature_all_pca_{i}.pth')
#     sub_features = features[node_idx]
#     torch.save(sub_features, f'feature_10M/feature_10M_pca_{i}.pth')
               
# print('wiki-10M has been done.')
# pdb.set_trace()




