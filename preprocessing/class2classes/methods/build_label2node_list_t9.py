
import torch
import pdb
from tqdm import tqdm

node2label = torch.load('./../../../LHGXLP/dataset/labels_full_32.pth')

num_classes = int(torch.max(node2label).item())+1
label2node_list = [[] for _ in range(num_classes)]
for n in tqdm(range(node2label.shape[0])):
    for l in range(node2label.shape[1]):
        if l != -1:
            label2node_list[node2label[n,l]].append(n)

torch.save(label2node_list, './../../../LHGXLP/dataset/label2node_list.pth')

