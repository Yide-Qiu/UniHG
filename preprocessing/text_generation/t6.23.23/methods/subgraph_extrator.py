import random
import torch
import pdb 
import pickle as pkl
import dgl
import numpy as np
from tqdm import tqdm
from utils import entity_select

# global device
# device = 'cuda'

def filter_minus_one_nodes(nodes):
    return [node for node in nodes if node != -1]

random.seed(123)

features = pkl.load(open('./../feature_all_pca.pkl', 'rb'))
labels = torch.load('./../labels_cluster.pth')
edge_idx_r = torch.load('./../relational_edge_index.pth')

# full graph
src_list = edge_idx_r[0].int()
dst_list = edge_idx_r[1].int()
graph = dgl.graph((src_list, dst_list))
graph.add_edges(torch.tensor([77312473], dtype=torch.int32), torch.tensor([77312473], dtype=torch.int32))
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
print(f'wiki-full has {num_nodes} nodes and {num_edges} edges.')
graph.ndata['feat'] = features

# pdb.set_trace()

# idx = list(range(features.shape[0]))
# random.shuffle(idx)
# num = 1000000
####################################################
# node_idx = random.sample(range(num_nodes), 1000000) # no random sample node because the small edges. select top-k degree nodes.
# sample_num = 1000000
# in_degrees = graph.in_degrees().numpy()
# topk_indices = np.argsort(in_degrees)[-int(sample_num/10):]
# fine = False
# base_length = 50
# while fine == False:
#     base_length += 10
#     walks, _= dgl.sampling.random_walk(graph, nodes=topk_indices, length=base_length, restart_prob=0.0)
#     tensor_set = set(walks.view(-1).tolist())
#     print(f'now the length is {base_length}')
#     if len(tensor_set) >= sample_num:
#         fine = True
#         node_idx = list(tensor_set)
# sub_graph = dgl.node_subgraph(graph, node_idx)
# print(f'wiki-1M has {sub_graph.number_of_nodes()} nodes and {sub_graph.number_of_edges()} edges.')
# sub_features = sub_graph.ndata['feat']
# sub_labels = labels[node_idx] # 
# print(sub_labels)
# pdb.set_trace()
# ## we need to contain the edge_index_1M and 10M. 
# edge_mapping = sub_graph.edata['_ID']
# edge_index_1M = sub_graph.edges(form='uv', order='eid')
# tensor1, tensor2 = edge_index_1M
# edge_index_1M = torch.cat((tensor1.unsqueeze(0), tensor2.unsqueeze(0)), dim=0)
# edge_index_1M = torch.cat([edge_index_1M, torch.zeros((1,edge_index_1M.shape[1]))], dim=0)
# for edge_sg_id in tqdm(range(edge_mapping.shape[0])):
#     edge_index_1M[2][edge_sg_id]=edge_idx_r[2][edge_mapping[edge_sg_id]]
# torch.save(sub_features, './../new_feature_1M_pca.pth')
# torch.save(sub_labels, './../new_labels_cluster_1M.pth')
# torch.save(edge_index_1M, './../edge_index_1M.pth')

# num_hops = 5
# for i in range(1, num_hops):
#     features = torch.load(f'./../feature_all_pca_{i}.pth')
#     sub_features = features[node_idx]
#     torch.save(sub_features, f'./../new_feature_1M_pca{i}.pth')

# print('wiki-1M has been done.')
# pdb.set_trace()

####################################################
sample_num = 10000000
in_degrees = graph.in_degrees().numpy()
topk_indices = np.argsort(in_degrees)[-int(sample_num/5):]
random_indices = random.sample(range(num_nodes), int(sample_num/2))
indices = list(topk_indices) + random_indices
fine = False
base_length = 90
while fine == False:
    walks, _= dgl.sampling.random_walk(graph, nodes=indices, length=base_length, restart_prob=0.0)
    # pdb.set_trace()
    # filtered_walks = torch.tensor(np.array([filter_minus_one_nodes(walk) for walk in tqdm(walks)]).flatten())
    tensor_set = set(walks.view(-1).tolist())
    print(f'now the length is {base_length} nodes : {len(tensor_set)}')
    if len(tensor_set) >= sample_num:
        fine = True
        node_idx = list(tensor_set)
    base_length += 10
# pdb.set_trace()
node_idx = [x for x in tqdm(node_idx) if x != -1]
sub_graph = dgl.node_subgraph(graph, node_idx)
print(f'wiki-10M has {sub_graph.number_of_nodes()} nodes and {sub_graph.number_of_edges()} edges.')
sub_features = sub_graph.ndata['feat']
sub_labels = labels[node_idx] 
print(sub_labels)
edge_mapping = sub_graph.edata['_ID']
edge_index_10M = sub_graph.edges(form='uv', order='eid')
tensor1, tensor2 = edge_index_10M
edge_index_10M = torch.cat((tensor1.unsqueeze(0), tensor2.unsqueeze(0)), dim=0)
edge_index_10M = torch.cat([edge_index_10M, torch.zeros((1,edge_index_10M.shape[1]))], dim=0)
for edge_sg_id in tqdm(range(edge_mapping.shape[0])):
    edge_index_10M[2][edge_sg_id]=edge_idx_r[2][edge_mapping[edge_sg_id]]
torch.save(sub_features, './../new_feature_10M_pca.pth')
torch.save(sub_labels, './../new_labels_cluster_10M.pth')
torch.save(edge_index_10M, './../edge_index_10M.pth')

num_hops = 5
for i in range(1, num_hops):
    features = torch.load(f'./../feature_all_pca_{i}.pth')
    sub_features = features[node_idx]
    torch.save(sub_features, f'./../new_feature_10M_pca_{i}.pth')
               
print('wiki-10M has been done.')
pdb.set_trace()




