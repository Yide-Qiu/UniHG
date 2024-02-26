import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成一些示例数据
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
indicators = ['Indicator 1', 'Indicator 2', 'Indicator 3', 'Indicator 4']
datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
results = np.random.rand(len(indicators), len(models), len(datasets)) * 100

# 设置样式
sns.set(style='whitegrid')

# 创建一个图表，包含多个子图
fig, axes = plt.subplots(1, len(datasets), figsize=(12 * len(datasets), 8 ))

# 遍历每个数据集
for d, dataset in enumerate(datasets):
    ax = axes[d]
    
    # 绘制热图
    sns.heatmap(results[:, :, d], annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=models, yticklabels=indicators, ax=ax)
    
    # 设置子图标题和标签
    ax.set_title(f'Results for {dataset}')
    ax.set_xlabel('Models')
    ax.set_ylabel('Indicators')

# 调整子图之间的间距
plt.tight_layout()

# 保存图表为 PNG 格式
plt.savefig('plot/experiment_1.png')

# 显示图表
# plt.show()
