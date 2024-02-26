import torch
import clip
import pdb
import pickle as pkl
from tqdm import tqdm
import torch
import torch.nn as nn
import re
import os

edge_id2label = pkl.load(open('./../../text_generation/t9.21.23/relationship_dict2.pkl', 'rb')) # 10697
edge_id2idx = pkl.load(open('./../../text_generation/t9.21.23/edge2idx_dict.pkl', 'rb')) # 2093+1

edge_text_list = []
for id in edge_id2idx:
    edge_text_list.append(edge_id2label[id])
edge_text_list.append('self-loop')

with open('./../edge_text_list.pkl', 'wb') as f:
    pkl.dump(edge_text_list, f)
    f.close()
