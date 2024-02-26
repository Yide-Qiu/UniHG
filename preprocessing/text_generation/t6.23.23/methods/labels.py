import pickle as pkl
import torch
import pdb

c_labels = torch.load('cluster_indices.pth')
labels = pkl.load(open('label_txt_list2.pkl','rb'))
pdb.set_trace()

