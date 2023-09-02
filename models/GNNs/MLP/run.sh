

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install torchmetrics

CUDA_VISIBLE_DEVICES=0 python src/GNNs/MLP/mlp.py --batch_size 300000

CUDA_VISIBLE_DEVICES=1 python src/GNNs/MLP/mlp.py --dataset wiki_1M --batch_size 20000

CUDA_VISIBLE_DEVICES=2 python src/GNNs/MLP/mlp.py --dataset wiki_10M --batch_size 100000


