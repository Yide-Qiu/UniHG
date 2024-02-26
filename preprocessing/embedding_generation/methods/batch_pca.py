import cupy as cp
import numpy as np
import pdb
import pickle as pkl
import torch

# 生成一个随机的100万乘700的矩阵作为示例数据，实际情况下，你需要用你自己的数据替换这里的随机矩阵
# np.random.seed(42)

def pca_gpu(data, n_components, batch_size):
    # 使用批处理，将数据拆分为小块，然后在GPU上逐块进行计算
    reduced_data_batches = []
    for i in range(num_batches):

        end = min(data.shape[0], (i + 1) * batch_size)
        data_batch = data[i * batch_size: end]

        # 将数据转换为CuPy数组，并将其发送到GPU
        data_gpu = cp.asarray(data_batch)

        # 中心化数据
        data_gpu -= cp.mean(data_gpu, axis=0)

        # 计算协方差矩阵
        cov_matrix_gpu = cp.dot(data_gpu.T, data_gpu) / (data_gpu.shape[0] - 1)

        # 使用CuPy进行特征值分解
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(cov_matrix_gpu)

        # 选择前n_components个特征向量
        top_eigenvectors_gpu = eigenvectors_gpu[:, -n_components:]

        # 降维操作
        reduced_data_gpu = cp.dot(data_gpu, top_eigenvectors_gpu)

        # 将降维后的数据从GPU转换回CPU
        reduced_data_batch = cp.asnumpy(reduced_data_gpu)
        reduced_data_batches.append(reduced_data_batch)

    reduced_data = np.concatenate(reduced_data_batches, axis=0)

    return reduced_data

# 生成示例数据
# data = np.random.rand(*data_shape)
# feature_file = "TAPE/feature_all.pkl"
# edge_feature_file = "TAPE/edge_feature.pkl"
labels_feature_file = "./../node_features_frozen.pkl"
labels_feature = pkl.load(open(labels_feature_file, 'rb'))
# labels_feature = torch.load(labels_feature_file) # [76k, 768]
print(labels_feature.shape)
data = labels_feature

batch_size = 100000  # 每次处理的样本数
global num_batches
num_batches = int(data.shape[0] / batch_size)+1
print(num_batches)



# 调用PCA进行降维，将700维降为128维
n_components = 128
# label降维
reduced_data = pca_gpu(data, n_components, batch_size)
reduced_data = torch.tensor(reduced_data)
# 打印降维后的数据维度
print("降维后的数据维度：", reduced_data.shape)
# torch.save(reduced_data, './../node_features_frozen_pca.pth')
torch.save(reduced_data, './../embeddings/edge_features_frozen_pca.pth')
pdb.set_trace()


