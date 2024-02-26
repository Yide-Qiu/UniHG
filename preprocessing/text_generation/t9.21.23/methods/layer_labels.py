import json
import time
import pdb
import dgl
import pickle as pkl
import torch
from tqdm import tqdm
from utils import entity_or_attribute, label_select, entity_select, is_attribute, is_entity, Bad_Attribute

# num_entities = 0
# num_subclass = 0
# num_labels = 0


# json_file = './../../../../rawdata/latest-all.json'
# label_id2des_dict = pkl.load(open('./../label_id2des_dict.pkl', 'rb'))
# label_id2idx_dict = pkl.load(open('./../label_id2idx_dict.pkl', 'rb'))
# src = []
# dst = []
# subclass_dict = {}

# pdb.set_trace()

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

#     if js['id'] in label_id2idx_dict and 'P279' in js['claims'].keys():
#         num_labels += 1
#         num_f_labels = len(js['claims']['P279'])
#         for claim_idx in range(num_f_labels): 
#             if 'datavalue' not in js['claims']['P279'][claim_idx]['mainsnak']:
#                 continue
#             num_subclass += 1
#             if js['claims']['P279'][claim_idx]['mainsnak']['datavalue']['value']['id'] in label_id2idx_dict:
#                 src.append(label_id2idx_dict[js['id']])
#                 dst.append(label_id2idx_dict[js['claims']['P279'][claim_idx]['mainsnak']['datavalue']['value']['id']])
#             else:
#                 src.append(label_id2idx_dict[js['id']])
#                 dst.append(js['claims']['P279'][claim_idx]['mainsnak']['datavalue']['value']['id'])

#     num_entities += 1
#     if num_entities % 1000 == 0:
#         print(f"Has done {num_entities} entities!")
#         print(f"There are {num_labels} labels!")
#         print(f"There are {num_subclass} subclasses!")

# pdb.set_trace()

# src_dst = {"src":src, "dst":dst}
# with open('./../src_dst.pkl', 'wb') as f: 
#     pkl.dump(src_dst, f)
#     f.close()


num_entities = 0
num_subclass = 0
num_labels = 0

src_dst = pkl.load(open('./../src_dst.pkl', 'rb'))
label_id2des_dict = pkl.load(open('./../label_id2des_dict.pkl', 'rb'))

subclass_dict = {}

for src,dst in zip(src_dst["src"],src_dst["dst"]):
    if type(dst) != int:
        continue
    num_subclass += 1
    if src not in subclass_dict:
        subclass_dict[src] = [dst]
    else:
        subclass_dict[src].append(dst)

def expand_dict(input_dict):
    expanded_dict = {}

    stack = list(input_dict.keys())
    visited = set()

    while stack:
        current_key = stack.pop()
        if current_key not in visited:
            visited.add(current_key)
            expanded_dict[current_key] = []

            for sub_key in input_dict.get(current_key, []):
                stack.append(sub_key)
                expanded_dict[current_key].extend([sub_key] + expanded_dict.get(sub_key, []))

    return expanded_dict

temp = {1:[2,3,4],2:[6,7],3:[8,9],6:[5]}
expanded_dict1 = expand_dict(temp)

expanded_dict2 = expand_dict(subclass_dict)
f = open('./../layered_labels.pkl', 'wb');pkl.dump(expanded_dict2, f);f.close()

layered_labels = pkl.load(open('./../layered_labels.pkl', 'rb'))
multilabels = torch.load('./../multilabels.pth')
label_entity_dict = pkl.load(open('./../label_entity_dict.pkl', 'rb'))

label = torch.ones(size=(multilabels.shape[0], 64)) * (-1)
label[:,:16] = multilabels

# def insert_values(tensor, dictionary):
#     for key, values in dictionary.items():
#         row_index = torch.nonzero((tensor[:, :len(values)] == key).all(dim=1), as_tuple=True)[0]
#         if len(row_index) > 0:
#             tensor[row_index, len(values):] = torch.tensor(values + [-1] * (tensor.size(1) - len(values)))
#     return tensor

# A = torch.tensor([[1, 5, 6, -1, -1, -1],[2, 8, -1, -1, -1, -1]])
# B = {1: [2, 3, 4], 5: [7]}
# A = insert_values(A, B)

# print(A)

# pdb.set_trace()

# label = insert_values(label, layered_labels)


pdb.set_trace()


