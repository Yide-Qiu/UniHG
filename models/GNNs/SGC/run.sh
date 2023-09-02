


pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install torchmetrics

CUDA_VISIBLE_DEVICES=3 python src/GNNs/SGC/sgc.py --use_sgc_embedding --batch_size 300000

CUDA_VISIBLE_DEVICES=4 python src/GNNs/SGC/sgc.py --use_sgc_embedding --dataset wiki_1M --batch_size 20000

CUDA_VISIBLE_DEVICES=5 python src/GNNs/SGC/sgc.py --use_sgc_embedding --dataset wiki_10M --batch_size 100000


