
import torch
import pdb 
import pickle as pkl
import dgl
import dgl.function as fn
import time
from tqdm import tqdm
import json
from utils import entity_select

edge_file = './../relative_adjacent3.txt'
edge_index_file = './../relational_edge_index.pth'
global num_edges
num_edges = 641738095
edge_index = torch.Tensor(size=(3, num_edges))
edge_id = 0
with open(edge_file, 'r') as f:
    for line in f:
        # pdb.set_trace()
        spl = line.split(" ")
        edge_index[0][edge_id] = int(spl[0])
        edge_index[1][edge_id] = int(spl[2][:-1])
        edge_index[2][edge_id] = int(spl[1])
        edge_id += 1
        if edge_id%10000 == 0:
            print(f"已建立{edge_id}条边")

torch.save(edge_index, edge_index_file)

# with open(edge_index_file, 'wb') as f:
#     pkl.dump(edge_index, f)
#     f.close()
pdb.set_trace()

# problem : 目前的问题在于，邻接阵中的节点只有77312473个，少了一个，这一个去哪了呢
# 猜测可能是因为最后一个节点它和前边的节点之间没有边，所以dgl.graph的时候被忽略了，如果确实如此，我们可以手动加上这个点
# 可以看entity2idx

# 然而 我们发现嫌疑点Q120043165具有多个应有的边

# 重新生成了带自环的relational edge index，应当已解决
# 仍未解决
# 直接补一个点吧

global device
device = 'cuda'

# entity_id = 0
# json_file = 'latest-all.json'
# json_all = open(json_file, 'r')
# for line in json_all:
#     l1 = len(line)
#     s1 = line
#     if l1 < 5:
#         continue
#     while s1[-1] != '}' :
#         l1 -= 1
#         s1 = line[:l1]
#     js = json.loads(s1)
#     id = js['id']
#     if id == 'Q120043165':
#         pdb.set_trace()
#     entity_id += 1
#     if entity_id % 100 == 0:
#         print(f'done {entity_id}')

feature = pkl.load(open('feature_all/feature_all_pca.pkl', 'rb'))
edge_idx_r = pkl.load(open('relational_edge_index3.pkl', 'rb'))
edge_feat = pkl.load(open('edge_feature/edge_feature_pca.pkl', 'rb'))
print(edge_feat.shape)
edge_feat = torch.cat((edge_feat,torch.ones((1,edge_feat.shape[1]))), dim=0)

src_list = edge_idx_r[0].int()
dst_list = edge_idx_r[1].int()
link_list = edge_idx_r[2].int()
link_list = torch.cat((link_list,torch.tensor([2082], dtype=torch.int32)), dim=0)


# nodes = list(set(src_list + dst_list))
# num_nodes = len(nodes)
# node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes)}
# src_list = [node_mapping[idx] for idx in src_list]
# dst_list = [node_mapping[idx] for idx in dst_list]

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



