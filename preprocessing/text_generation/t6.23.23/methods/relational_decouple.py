
import torch
import pickle as pkl

feature = pkl.load(open('feature_all/feature_all_pca.pkl', 'rb'))
edge_idx_r = pkl.load(open('relative_edge_index.pkl', 'rb'))
edge_feat = pkl.load(open('edge_feature/edge_feature_pca.pkl', 'rb'))

# need a dgl graph to message passing





















