
import torch
import pdb
# from sklearn.datasets import make_blobs
from tqdm import tqdm   
from sklearn.preprocessing import StandardScaler

# 生成示例数据
# data, _ = make_blobs(n_samples=70000, centers=5, n_features=128, random_state=42)
# data = torch.tensor(data, dtype=torch.float32)

wiki_labels = torch.load('labels_emb_pca.pth') # [76k, 128]
data = wiki_labels

# pdb.set_trace()

# 归一化数据
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 设置聚类的类别数
num_clusters = 2000

# 将数据移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.tensor(data).to(device)

# 随机初始化聚类中心
centroids = data[:num_clusters].clone()

# 迭代次数
num_iterations = 100

for _ in range(num_iterations):
    # 计算每个样本与聚类中心的距离
    distances = torch.cdist(data, centroids)

    # 根据距离将样本分配到最近的聚类中心
    _, cluster_indices = torch.min(distances, dim=1)

    # 更新聚类中心为每个簇的均值
    for i in range(num_clusters):
        cluster_data = data[cluster_indices == i]
        if len(cluster_data) > 0:
            centroids[i] = cluster_data.mean(dim=0)

print("聚类完成！")

# 将聚类后的类别从GPU移回CPU
cluster_indices = cluster_indices.cpu().numpy()

# 输出每个特征聚类后的类别
print(cluster_indices) # [76k, 1]
torch.save(cluster_indices, 'cluster_indices.pth')

# 根据cluster_indices做标签映射
mutilabels = torch.load('multilabels/labels.pth') # [7700w,16] -1 padding
print(mutilabels)

for i in tqdm(range(mutilabels.size(0))):
    for j in range(mutilabels.size(1)):
        element = mutilabels[i, j]
        if element != -1:
            mutilabels[i, j] = cluster_indices[mutilabels[i, j]]
torch.save(mutilabels, 'multilabels/labels_cluster.pth')

pdb.set_trace()























