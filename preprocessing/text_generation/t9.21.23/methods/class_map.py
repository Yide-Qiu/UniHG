import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import pdb
import re
import random
import numpy as np
from tqdm import tqdm

def plot_nx(edge_index, text_list):
    G = nx.DiGraph()
    # edge_index = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    edge_iterator = edge_index.t()
    edge_list = []
    for src, dst in edge_iterator:
        edge_list.append((int(src.item()),int(dst.item())))
    G.add_edges_from(edge_list)
    print(G)
    # des_dic = {1: 'Node 1', 2: 'Node 2', 3: 'Node 3', 4: 'Node 4', 5: 'Node 5'}
    des_dic = {}
    id = 0
    for t in tqdm(text_list):
        des_dic[id] = t
        id += 1
    node_labels = des_dic

    # pdb.set_trace()
    degrees = G.degree()
    n = 1000
    top_n_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)[:n]
    pdb.set_trace()
    for idx,degree in tqdm(top_n_nodes):
        subgraph = nx.ego_graph(G, idx, radius=3, center=True)
        sample_ratio = 1
        if subgraph.number_of_nodes() > 1000:
            sample_ratio = 0.2
        elif subgraph.number_of_nodes() > 500:
            sample_ratio = 0.5
        # elif subgraph.number_of_nodes() > 100:
        #     sample_ratio = 0.5
        num_nodes_to_sample = int(sample_ratio * subgraph.number_of_nodes())
        sampled_nodes = random.sample(subgraph.nodes(), num_nodes_to_sample)
        sampled_graph = subgraph.subgraph(sampled_nodes)
        sub_nodes = list(sampled_graph.nodes())
        plt.figure(idx+1, figsize=(8, 8))
        pos = nx.spring_layout(sampled_graph)
        sampled_label = {}
        
        sv_text_list = list(np.array(text_list)[sub_nodes])
        for i in range(len(sub_nodes)):
            sampled_label[sub_nodes[i]] = node_labels[sub_nodes[i]]
        nx.draw(sampled_graph, pos, font_size=8, node_size=20, edge_color='blue', alpha=0.5)
        nx.draw_networkx_labels(sampled_graph, pos, labels=sampled_label, font_size=8)
        
        plt.title(f"Third Order Subgraph {idx+1}")
        plt.savefig(f'./../images/layered_class_{node_labels[idx]}.png')
        text = open(f'./../images/layered_class_{node_labels[idx]}.txt','w')
        for item in sv_text_list:
            text.write(str(item) + '\n')
        text.close()
        plt.close()
        # pdb.set_trace()


    # max_degree_node = max(G.nodes(), key=G.degree)
    # first_order_subgraph = nx.ego_graph(G, max_degree_node, radius=1)
    # second_order_subgraphs = [nx.ego_graph(G, node, radius=1) for node in first_order_subgraph.nodes()]
    # third_order_subgraphs = [nx.ego_graph(G, node, radius=1) for subgraph in second_order_subgraphs for node in subgraph.nodes()]
    # for i, subgraph in enumerate(third_order_subgraphs[:5]):
    #     plt.figure(i+1, figsize=(8, 8))
    #     nx.draw(subgraph, with_labels=True, font_size=8, node_size=20)
    #     plt.title(f"Third Order Subgraph {i+1}")
    #     plt.savefig(f'./../images/layered_class_{i}.png')

    # num_sub = 100
    # num_edges = G.number_of_edges()
    # sub_num_edges = int(num_edges/num_sub)
    # all_edges = list(G.edges())
    # for s in tqdm(range(num_sub)):
    #     sub_edges = all_edges[s*sub_num_edges : min((s+1)*sub_num_edges,num_edges)]
    #     subgraph = G.edge_subgraph(sub_edges)
    #     sub_nodes = list(subgraph.nodes())
    #     pdb.set_trace()
    #     original_graph_indices = [G.nodes[node]['index'] for node in sub_nodes]
    #     sub_labels = node_labels[original_graph_indices]
    #     pos = nx.spring_layout(subgraph)
    #     fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)  # 设置 figsize 和 dpi
    #     nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, edge_color='gray', ax=ax)
    #     nx.draw_networkx_labels(subgraph, pos, labels=sub_labels, font_size=8, font_color='red')
    #     plt.savefig(f'./../images/layered_class_{s}.png')

    # num_sub = 100
    # num_nodes = G.number_of_nodes()
    # sub_num_nodes = int(num_nodes/num_sub)
    # all_nodes = list(G.nodes())
    # for s in tqdm(range(num_sub)):
    #     sub_edges = all_edges[s*sub_num_edges : min((s+1)*sub_num_edges,num_edges)]
    #     subgraph = G.edge_subgraph(sub_edges)
    #     sub_nodes = list(subgraph.nodes())
    #     pdb.set_trace()
    #     original_graph_indices = [G.nodes[node]['index'] for node in sub_nodes]
    #     sub_labels = node_labels[original_graph_indices]
    #     pos = nx.spring_layout(subgraph)
    #     fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)  # 设置 figsize 和 dpi
    #     nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, edge_color='gray', ax=ax)
    #     nx.draw_networkx_labels(subgraph, pos, labels=sub_labels, font_size=8, font_color='red')
    #     plt.savefig(f'./../images/layered_class_{s}.png')

# pdb.set_trace()
layered_label_class2class = pkl.load(open('./../layered_label_class2class.pkl','rb')) # dic
label_id2idx = pkl.load(open('./../label_id2idx_dict.pkl','rb'))
labels_text_list = pkl.load(open('./../labels_text_list.pkl','rb'))
re_text_list = []
num_labels = len(label_id2idx)
pattern = re.compile(r'(.*?) be')

for t in labels_text_list:
    match = pattern.search(t)
    if match:
        extracted_text = match.group(1)
        print(extracted_text)
        re_text_list.append(extracted_text)
    else:
        re_text_list.append(t)
        print(t)
        # print("No match found.")    
# edge_index = torch.stack([torch.arange(num_labels),torch.arange(num_labels)])
edge_index = torch.Tensor()
for dst in tqdm(layered_label_class2class.keys()):
    src_list = layered_label_class2class[dst]
    for src in src_list:
        edge_index = torch.cat((edge_index, torch.tensor([[src],[dst]])), dim=1)
plot_nx(edge_index, re_text_list)




