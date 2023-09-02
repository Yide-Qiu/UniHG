import cupy as cp
import numpy as np
import pdb
import pickle as pkl
import torch


def pca_gpu(data, n_components, batch_size):
    reduced_data_batches = []
    for i in range(num_batches):
        end = min(data.shape[0], (i + 1) * batch_size)
        data_batch = data[i * batch_size: end]
        data_gpu = cp.asarray(data_batch)
        data_gpu -= cp.mean(data_gpu, axis=0)
        cov_matrix_gpu = cp.dot(data_gpu.T, data_gpu) / (data_gpu.shape[0] - 1)
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(cov_matrix_gpu)
        top_eigenvectors_gpu = eigenvectors_gpu[:, -n_components:]
        reduced_data_gpu = cp.dot(data_gpu, top_eigenvectors_gpu)
        reduced_data_batch = cp.asnumpy(reduced_data_gpu)
        reduced_data_batches.append(reduced_data_batch)

    reduced_data = np.concatenate(reduced_data_batches, axis=0)

    return reduced_data


labels_feature_file = "/data4/LightGCN-PyTorch/item_embedding_citeulike-a.pth"
labels_feature = torch.load(labels_feature_file) # [76k, 768]
batch_size = 100000  # 
global num_batches
num_batches = int(labels_feature.shape[0] / batch_size)+1
print(num_batches)

n_components = 128
reduced_data = pca_gpu(labels_feature, n_components, batch_size)
reduced_data = torch.tensor(reduced_data)
print(reduced_data.shape)
torch.save(reduced_data, '/data4/LightGCN-PyTorch/item_embedding_citeulike-a_pca128.pth')
pdb.set_trace()