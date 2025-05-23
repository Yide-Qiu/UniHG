# 🌐 UniHG: Universal Heterogeneous Graph Toolkit [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Official implementation of "[UniHG: A Large-scale Universal Heterogeneous Graph Dataset and Benchmark for Representation Learning and Cross-Domain Transfering](https://anonymous.4open.science/r/UniHG-AA78)"*

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

# 🚀 Highlights
✅ **Largest Universal Heterogeneous Graph Dataset**  
77.31M nodes | 564M edges | 2,082 relation types | 74K+ labels  
✅ **SOTA Performance**  
28.93% accuracy improvement | 22.1× faster training  
✅ **Cross-Domain Transfer**  
11.71% NDCG@20 boost in recommendation tasks  
✅ **Novel Framework**  
HGD with Anisotropic Feature Propagation (AFP)

# 📚 UniHG Introduction

https://yide-qiu.github.io/Pages_UniHG_Dataset/

# 🛠️ Quick Start

## Requirements:
```
torch                         2.0.1+cu117
numpy                         1.21.5
optuna                        3.5.0
scikit-learn                  1.0.2
scipy                         1.7.3
dgl                           1.1.1+cu117
torch-cluster                 1.6.1
torch-geometric               2.3.1
torch-scatter                 2.1.1
torch-sparse                  0.6.17
torch-spline-conv             1.2.2
networkx                      3.1
ogb                           1.3.6
nltk                          3.8.1
cupy                          12.2.0
```
## Download UniHG：

### UniHG-1M:
This is the smallest version (489.7MB × 5) of **UniHG**. It has 1,002,988 nodes with 46 types and 24,475,405 edges with 178 types. The dimension of node feature is 128. We have provided its 5-hop feature propagation matrixes to facilitate learning using decoupled graph neural networks. You can find **UniHG-1M** at [link](https://pan.quark.cn/s/fcf6c2ae7554).

### UniHG-10M:
This is a medium-sized version (4.79GB × 5) of **UniHG**. It has 10,044,777 nodes with 315 types and 216,295,022 edges with 729 types. The dimension of node feature is 128. We have provided its 5-hop feature propagation matrixes to facilitate learning using decoupled graph neural networks. You can find **UniHG-10M** at [link](https://pan.quark.cn/s/128a3c656005).

### UniHG-Full:
This is the largest version (36.87GB × 5) of **UniHG**. It has 77,312,474 nodes with 2,000 types and 641,738,096 edges with 2,082 types. The dimension of node feature is 128. We have provided its 5-hop feature propagation matrixes to facilitate learning using decoupled graph neural networks. You can find **UniHG-Full** at [link](https://pan.quark.cn/s/252cf3117451).

# 🧠 Framework Architecture

## Pipeline
This work focuses on constructing the largest universal domain heterogeneous graph available and effectively learning its representation as well as transferring the universal knowledge to other downstream graph task. The overall task pipeline is shown below:

![Alt](./figs/pipeline.png)

## Preprocessing:

We use the **JSON** version of all wikidata data from October 23, 2024 to form our dataset.
See preprocessing for more details on our processing strategy.

## Datasets:
UniHG is a universal dataset compared to other isolated datasets. This means that there are "bridges" in UniHG that connect these isolated datasets. Naturally, UniHG also has more types of nodes and edges. The visualization of the generic dataset is illustrated in the figure below:
![Alt](./figs/diff.png)
We further evaluated UniHG using multiple metrics, as visualized in the following figure:
![Alt](./figs/metric.png)

## How to construct UniHG?
We have mapped the overall flow of the composition:
![Alt](./figs/construct_graph.png)

## Training:
Further, in order to efficiently learn complex representations of UniHG, we propose a new representation learning framework **HGD** (Heterogeneous Graph Decoupling Framework). To evaluate the effectiveness of **HGD**, we compare other sampling-based convolutional type graph neural networks (**GCN**, **HAN**, **HGT**) and decoupling-based graph neural networks (**SGC**, **SIGN**, **GAMLP**) on three sizes of UniHG datasets. We used official implementations of these methods.
The training commands are detailed in the individual run.sh files in the models.

# 📊 Benchmark Results

Results of comparison experiments on **UniHG-1M**, **UniHG-10M**, and **UniHG-full**:

![Alt](./figs/comparison_experiment.png)

Results of knowledge transfer experiments on recommendation system:

![results_of_knowledge_transfer_experiments](./figs/transfer.png)

Results of the ablation experiments on **UniHG-1M**, **UniHG-10M**, and **UniHG-full**. '-AFP' means 'using the feature of Anisotropic Feature Propagation'.

![](./figs/ablation.png)


# 📜 Citation

@article{unihg2025,
  title={UniHG: A Large-scale Universal Heterogeneous Graph Dataset and Benchmark for Representation Learning and Cross-Domain Transfering},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}
