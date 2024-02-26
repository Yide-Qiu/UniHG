import pickle
import torch

with open('label_txt_list2.pkl', 'rb') as f:
    label_txt = pickle.load(f)

cluster_indices = torch.load('cluster_indices.pth')

org2cluster = [[] for i in range(max(cluster_indices) + 1)]
for idx, new_type in enumerate(cluster_indices):
    print(idx)
    org2cluster[new_type].append(idx)

for item in org2cluster:
    # print(label_txt[item])
    for org in item:
        print(label_txt[org])