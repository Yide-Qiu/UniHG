# REBUTTAL FOR ICML 2025

# session1 
CUDA_VISIBLE_DEVICES=4 python UniHG-main/models/GNNs/GAMLP/gamlp.py --use_gamlp_embedding --dataset wiki_1M --batch_size 20000 
# session7 
CUDA_VISIBLE_DEVICES=0 python UniHG-main/models/GNNs/GAMLP/gamlp.py --use_gamlp_embedding --dataset wiki_10M --batch_size 70000 --epochs 200

# session2  
CUDA_VISIBLE_DEVICES=4 python UniHG-main/models/GNNs/HGD/hgd.py --use_hgd_embedding --dataset wiki_1M --batch_size 20000 
# session8
CUDA_VISIBLE_DEVICES=2 python UniHG-main/models/GNNs/HGD/hgd.py --use_hgd_embedding --dataset wiki_10M --batch_size 70000 --epochs 200

# session3 
CUDA_VISIBLE_DEVICES=4 python UniHG-main/models/GNNs/SGC/sgc.py --use_sgc_embedding --dataset wiki_1M --batch_size 20000 
# session9 
CUDA_VISIBLE_DEVICES=2 python UniHG-main/models/GNNs/SGC/sgc.py --use_sgc_embedding --dataset wiki_10M --batch_size 70000 --epochs 200

# session4 
CUDA_VISIBLE_DEVICES=4 python UniHG-main/models/GNNs/SIGN/sign.py --use_sign_embedding --dataset wiki_1M --batch_size 20000 
# session10
CUDA_VISIBLE_DEVICES=4 python UniHG-main/models/GNNs/SIGN/sign.py --use_sign_embedding --dataset wiki_10M --batch_size 70000 --epochs 200

Dataset	Method	Seed	Accuracy	Precision	F1	TrainTime(s)	Space(MB)
wiki_1M	SGC	5678	44.76%	85.86%	58.28%	29.37104821205139	1692.37890625
wiki_1M	SGC	8901	45.77%	86.33%	58.96%	28.563913822174072	1897.02734375
wiki_1M	SGC	2345	43.65%	87.78%	57.18%	27.526328563690186	1647.7890625
wiki_1M	GAMLP	4567	47.45%	83.67%	60.46%	30.42868661880493	3740.66015625
wiki_10M	SIGN	5678	47.14%	89.08%	61.77%	218.9333724975586	21832.34375
wiki_1M	SIGN	6789	62.03%	94.19%	73.70%	32.74782633781433	3300.89453125
wiki_1M	SIGN	5678	58.09%	93.84%	70.38%	31.072935342788696	2988.3828125
wiki_1M	GAMLP	2345	47.57%	83.17%	60.51%	30.406272172927856	4774.70703125
wiki_1M	SIGN	8901	62.00%	95.38%	73.85%	30.539133548736572	3319.9453125
wiki_10M	GAMLP	3456	59.71%	83.16%	71.70%	178.25947761535645	29020.421875
wiki_1M	GAMLP	9012	47.25%	86.54%	60.66%	49.547467947006226	4405.34375
wiki_10M	GAMLP	5678	62.35%	84.30%	74.01%	235.88018155097961	29126.06640625
wiki_1M	GAMLP	8901	48.99%	86.65%	62.34%	32.349517822265625	3733.66015625
wiki_10M	SGC	7890	63.67%	93.49%	76.39%	203.8823447227478	7137.1171875
wiki_1M	GAMLP	1234	47.25%	84.58%	60.29%	36.354591608047485	3899.390625
wiki_10M	HGD	1234	88.69%	96.92%	93.18%	340.55385184288025	26912.625
wiki_1M	SIGN	1123	72.27%	90.01%	80.24%	42.53747844696045	2966.3125
wiki_1M	GAMLP	6789	48.73%	84.61%	61.44%	28.961890935897827	4572.6015625
wiki_1M	SIGN	2345	62.83%	94.95%	74.39%	31.1074435710907	3319.421875
wiki_1M	GAMLP	5678	49.12%	83.78%	61.78%	42.100457429885864	3557.00390625
wiki_10M	HGD	567	88.52%	97.35%	93.15%	246.54427218437195	27072.6796875
wiki_1M	SIGN	3456	62.39%	94.02%	74.22%	36.97497534751892	2988.16015625
wiki_10M	SGC	1234	64.79%	93.11%	77.13%	278.12772727012634	7731.44140625
wiki_1M	SGC	6789	44.98%	85.95%	58.36%	26.51474928855896	1674.34765625
wiki_1M	HGD	901	75.43%	89.84%	82.47%	30.4120090007782	4392.171875
wiki_1M	SIGN	7890	34.04%	91.82%	48.45%	38.97712588310242	2986.5078125
wiki_1M	SGC	3456	43.98%	87.66%	57.74%	35.24685096740723	1567.98828125
wiki_10M	GAMLP	7890	58.23%	83.79%	71.33%	199.7873513698578	29022.375
wiki_1M	HGD	567	75.74%	88.38%	82.30%	49.11087942123413	4454.2890625
wiki_10M	SIGN	3456	80.15%	97.48%	88.25%	212.7792627811432	21740.30859375
wiki_1M	HGD	1234	75.54%	88.08%	82.16%	37.35831665992737	4948.68359375
wiki_1M	GAMLP	3456	46.68%	83.91%	59.52%	38.81470203399658	3651.734375
wiki_10M	SGC	9012	64.00%	92.62%	76.38%	215.99757981300354	7269.1484375
wiki_10M	SIGN	1234	57.81%	96.49%	72.57%	232.57477593421936	21935.1328125
wiki_1M	HGD	2025	75.74%	91.11%	83.03%	71.27836036682129	3784.925781256875
wiki_1M	SGC	1234	44.62%	87.40%	58.23%	28.75255823135376	1812.17578125
wiki_10M	GAMLP	1234	61.66%	83.90%	73.46%	298.75683426856995	29367.12890625
wiki_1M	SIGN	4567	60.23%	94.45%	72.14%	36.150065183639526	3312.7265625
wiki_1M	SGC	1123	43.71%	87.74%	57.63%	32.2597382068634	1355.03515625
wiki_1M	GAMLP	7890	47.83%	86.08%	60.84%	43.24765706062317	3545.984375
wiki_1M	SIGN	1234	70.37%	93.05%	79.76%	34.42877912521362	3278.8671875
wiki_10M	SGC	5678	63.28%	93.27%	76.11%	241.78229594230652	7586.72265625
wiki_10M	GAMLP	9012	58.10%	90.57%	71.51%	227.0588343143463	29102.13671875
wiki_1M	SGC	4567	45.38%	79.85%	58.39%	32.11145639419556	1527.2265625
wiki_10M	SIGN	9012	63.44%	96.33%	77.11%	163.63866090774536	21745.94140625
wiki_1M	GAMLP	1123	48.07%	85.46%	61.29%	45.16713809967041	4456.9296875
wiki_1M	SIGN	9012	62.55%	95.19%	74.25%	42.24449443817139	2987.10546875
wiki_10M	SGC	3456	61.77%	92.38%	74.85%	154.93898749351501	7169.72265625
wiki_1M	SGC	7890	44.76%	85.35%	58.02%	35.975473403930664	1354.8359375
wiki_1M	SGC	9012	46.36%	85.32%	59.43%	32.42104697227478	1689.99609375



# session5 new baseline MTMP 1M 
# stage1 Test  Acc: 0.44109339809417725%, precision: 0.8476009964942932%, recall: 0.4319688081741333%, f1: 0.5722383260726929
# stage2 Test  Acc: 0.46697012877464294%, precision: 0.8559020757675171%, recall: 0.4610541164875030%, f1: 0.5992316603660583
# stage3 Test  Acc: 0.47275748927346732%, precision: 0.8499238492462781%, recall: 0.4678127110492389%, f1: 0.6053213789127381
# stage4 Test  Acc: 0.47547873854637146%, precision: 0.8438768982887268%, recall: 0.4745171368122101%, f1: 0.6074362993240356
CUDA_VISIBLE_DEVICES=0 python UniHG-main/models/GNNs/GAMLP/mtmp.py --use_gamlp_embedding --dataset wiki_1M --batch_size 20000 --num_stages 4 --num_epochs 100



# session6 new baseline MTMP 10M 
# stage1 Test  Acc: 0.57468158006668091%, precision: 0.8474910259246826%, recall: 0.5977154373412262%, f1: 0.7010179758071899
# stage2 Test  Acc: 0.58302139081741124%, precision: 0.8714473761914721%, recall: 0.6058123804478932%, f1: 0.7126126784571641
# stage3 Test  Acc: 0.59323461368744561%, precision: 0.8833127389164672%, recall: 0.6103192867654126%, f1: 0.7214389127569153
# stage4 Test  Acc: 0.59926408529281622%, precision: 0.8952932953834534%, recall: 0.6138049363408996%, f1: 0.7282944917678833
CUDA_VISIBLE_DEVICES=0 python UniHG-main/models/GNNs/GAMLP/mtmp.py --use_gamlp_embedding --dataset wiki_10M --batch_size 70000 --num_stages 4 --num_epochs 100
# CUDA_VISIBLE_DEVICES=0 python UniHG-main/models/GNNs/GAMLP/mtmp.py --use_gamlp_embedding --dataset wiki_full --batch_size 200000 --num_stages 4 --num_epochs 100



# session11 Transfer to DeeperGCN on ogbl_collab 
# 24G OOM
cd ../deep_gcns_torch_master/examples/ogb/ogbl_collab
CUDA_VISIBLE_DEVICES=7 python main.py --use_gpu --num_layers 7 --block res+ --gcn_aggr softmax --learn_t --t 1.0
CUDA_VISIBLE_DEVICES=5 python main.py --UniHG --use_gpu --num_layers 7 --block res+ --gcn_aggr softmax --learn_t --t 1.0



# session13 Transfer to Topolink on ogbl_collab 
CUDA_VISIBLE_DEVICES=7 python main.py --dataset_name ogbl-collab --year 2007 --num_workers 16 --lr 0.00005 --batch_size 32 --hidden_channels 64 --out_mlp_dim 128 --num_seal_layers 2 --mlp_dropout 0.5 --num_hops 1 --use_feature 1 --use_ph 1 --pi_dim 25 --vit_dim 32 --vit_out_dim 64 --vit_depth 4 --vit_headers 4 --vit_mlp_dim 64 --extend 1 --use_pe True --walk_length 32 --epochs 20 --dynamic_train --dynamic_val --dynamic_test --train_samples 0.15
CUDA_VISIBLE_DEVICES=7 python main.py --UniHG --dataset_name ogbl-collab --year 2007 --num_workers 16 --lr 0.00005 --batch_size 32 --hidden_channels 64 --out_mlp_dim 128 --num_seal_layers 2 --mlp_dropout 0.5 --num_hops 1 --use_feature 1 --use_ph 1 --pi_dim 25 --vit_dim 32 --vit_out_dim 64 --vit_depth 4 --vit_headers 4 --vit_mlp_dim 64 --extend 1 --use_pe True --walk_length 32 --epochs 20 --dynamic_train --dynamic_val --dynamic_test --train_samples 0.15



# session14 Transfer to PLNLP on ogbl_collab and ogbl_ddi 
CUDA_VISIBLE_DEVICES=6 python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --epochs=800 --eval_last_best=True --dropout=0.3
python main.py --UniHG --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --epochs=800 --eval_last_best=True --dropout=0.3
CUDA_VISIBLE_DEVICES=6 python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3 
python main.py --UniHG --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3 



# session12 Transfer to BUDDY on [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi] 
cd subgraph-sketching/src
# Epoch: 99, Best epoch: 43, Loss: 0.0018, Train: 100.00%, Valid: 68.39%, Test: 68.49%, epoch time: 92.4
CUDA_VISIBLE_DEVICES=7 nohup python runners/run.py --dataset_name ogbl-collab --year 2007 --model BUDDY > output_collab_buddy.txt 2>&1 &
# 0.38 Epoch: 99, Best epoch: 93, Loss: 0.8429, Train: 100.00%, Valid: 67.94%, Test: 68.97%, epoch time: 388.9
CUDA_VISIBLE_DEVICES=6 nohup python runners/run.py --UniHG --dataset_name ogbl-collab --year 2007 --model BUDDY > ./output_unihg/output_collab_buddy.txt 2>&1 &

# Epoch: 99, Best epoch: 18, Loss: 0.0004, Train: 100.00%, Valid: 91.11%, Test: 84.50%, epoch time: 1.0
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model BUDDY  
# 0.92 Epoch: 99, Best epoch: 25, Loss: 3.2235, Train: 100.00%, Valid: 90.71%, Test: 85.39%, epoch time: 1.9
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Cora --model BUDDY > ./output_unihg/output_cora_buddy.txt 2>&1 &

# Epoch: 99, Best epoch: 21, Loss: 0.0006, Train: 100.00%, Valid: 97.00%, Test: 89.52%, epoch time: 1.1
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Citeseer --model BUDDY
# 0.67 Epoch: 99, Best epoch: 8, Loss: 3.8289, Train: 100.00%, Valid: 96.73%, Test: 91.02%, epoch time: 1.5
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Citeseer --model BUDDY > ./output_unihg/output_Citeseer_buddy.txt 2>&1 &

# Epoch: 99, Best epoch: 23, Loss: 0.0003, Train: 100.00%, Valid: 84.07%, Test: 69.56%, epoch time: 6.0
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Pubmed --model BUDDY
# 0.26 Epoch: 99, Best epoch: 34, Loss: 0.9177, Train: 100.00%, Valid: 84.14%, Test: 72.56%, epoch time: 10.2
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Pubmed --model BUDDY > ./output_unihg/output_Pubmed_buddy.txt 2>&1 &

# Epoch: 149, Best epoch: 121, Loss: 0.0747, Train: 78.71%, Valid: 69.86%, Test: 79.62%, epoch time: 41.7
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --cache_subgraph_features --batch_size 131072 --model BUDDY > output_ddi_buddy.txt 2>&1 &
# 0.21 Epoch: 149, Best epoch: 143, Loss: 0.1629, Train: 75.12%, Valid: 69.29%, Test: 79.97%, epoch time: 59.7
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --cache_subgraph_features --batch_size 131072 --model BUDDY > ./output_unihg/output_ddi_buddy.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name ogbl-ppa --model BUDDY > output_ppa_buddy.txt 2>&1 & 



# # session15 BUDDY 代码族 SEALDGCNN Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
# conda activate buddy 
# cd /gpu-data/qyd_qt/subgraph-sketching-main/src
# # 22G OOM
# CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model SEALDGCNN  
# # 22G OOM
# CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Citeseer --model SEALDGCNN 
# # 22G OOM
# CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Pubmed --model SEALDGCNN
# #
# CUDA_VISIBLE_DEVICES=7 nohup python runners/run.py --dataset_name ogbl-collab --year 2007 --model SEALDGCNN > output_collab_SEALDGCNN.txt 2>&1 &
# # CUDA_VISIBLE_DEVICES=5 nohup python runners/run.py --dataset_name ogbl-ppa --model SEALDGCNN > output_ppa_SEALDGCNN.txt 2>&1 &
# # 200h
# CUDA_VISIBLE_DEVICES=5 nohup python runners/run.py --dataset ogbl-ddi --sign_k 2 --model SEALDGCNN > output_ddi_SEALDGCNN.txt 2>&1 &



# session16 BUDDY ELPH Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
conda activate buddy 
cd /gpu-data/qyd_qt/subgraph-sketching-main/src
# Epoch: 99, Best epoch: 13, Loss: 0.0031, Train: 100.00%, Valid: 93.28%, Test: 86.28%, epoch time: 2.5
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model ELPH  
# 0.12 Epoch: 99, Best epoch: 10, Loss: 0.9888, Train: 100.00%, Valid: 93.28%, Test: 86.38%, epoch time: 4.9
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Cora --model ELPH > ./output_unihg/output_cora_ELPH.txt 2>&1 &

# Epoch: 99, Best epoch: 36, Loss: 0.0056, Train: 100.00%, Valid: 96.19%, Test: 88.80%, epoch time: 2.9
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Citeseer --model ELPH 
# 0.11 Epoch: 99, Best epoch: 19, Loss: 1.0055, Train: 100.00%, Valid: 95.91%, Test: 89.39%, epoch time: 2.8
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Citeseer --model ELPH > ./output_unihg/output_Citeseer_ELPH.txt 2>&1 &

# Epoch: 99, Best epoch: 22, Loss: 0.0073, Train: 100.00%, Valid: 84.36%, Test: 73.56%, epoch time: 17.5
CUDA_VISIBLE_DEVICES=0 nohup python runners/run.py --dataset_name Pubmed --model ELPH  > output_Pubmed_ELPH.txt 2>&1 &
# 0.12 Epoch: 99, Best epoch: 20, Loss: 0.9646, Train: 100.00%, Valid: 83.91%, Test: 73.42%, epoch time: 27.2
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Pubmed --model ELPH  > ./output_unihg/output_Pubmed_ELPH.txt 2>&1 &

# 24G OOM
CUDA_VISIBLE_DEVICES=7 nohup python runners/run.py --dataset_name ogbl-collab --year 2007 --model ELPH > output_collab_ELPH.txt 2>&1 &
# 24G OOM
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name ogbl-collab --year 2007 --model ELPH > ./output_unihg/output_collab_ELPH.txt 2>&1 &

# Epoch: 99, Best epoch: 6, Loss: 0.3542, Train: 27.74%, Valid: 26.57%, Test: 31.64%, epoch time: 495.2
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset ogbl-ddi --sign_k 2 --model ELPH > output_ddi_ELPH.txt 2>&1 &
# 0.14 Epoch: 99, Best epoch: 4, Loss: 1.3367, Train: 26.72%, Valid: 26.49%, Test: 31.99%, epoch time: 696.7
CUDA_VISIBLE_DEVICES=0 nohup python runners/run.py --UniHG --dataset ogbl-ddi --sign_k 2 --model ELPH > ./output_unihg/output_ddi_ELPH.txt 2>&1 &



# session17 BUDDY SEALGIN Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
conda activate buddy 
cd /gpu-data/qyd_qt/subgraph-sketching-main/src
# Epoch: 99, Best epoch: 15, Loss: 0.0298, Train: 53.13%, Valid: 73.72%, Test: 72.26%, epoch time: 13.1
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model SEALGIN  
# 0.18 Epoch: 99, Best epoch: 5, Loss: 0.1589, Train: 91.94%, Valid: 72.92%, Test: 72.56%, epoch time: 19.4
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Cora --model SEALGIN > ./output_unihg/output_Cora_SEALGIN.txt 2>&1 &

# Epoch: 99, Best epoch: 42, Loss: 0.1283, Train: 98.91%, Valid: 81.20%, Test: 75.33%, epoch time: 10.8
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Citeseer --model SEALGIN 
# 0.29 Epoch: 99, Best epoch: 42, Loss: 0.4191, Train: 95.11%, Valid: 79.56%, Test: 76.19%, epoch time: 24.7
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --UniHG --dataset_name Citeseer --model SEALGIN > ./output_unihg/output_Citeseer_SEALGIN.txt 2>&1 &

# Epoch: 99, Best epoch: 73, Loss: 0.2292, Train: 52.21%, Valid: 69.13%, Test: 64.02%, epoch time: 35.1
CUDA_VISIBLE_DEVICES=7 nohup python runners/run.py --dataset_name Pubmed --model SEALGIN > output_Pubmed_SEALGIN.txt 2>&1 &
# 0.16 Epoch: 99, Best epoch: 65, Loss: 0.3227, Train: 48.41%, Valid: 69.49%, Test: 65.25%, epoch time: 42.4
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Pubmed --model SEALGIN > ./output_unihg/output_Pubmed_SEALGIN.txt 2>&1 &



# session18 BUDDY SEALGCN Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
conda activate buddy 
cd /gpu-data/qyd_qt/subgraph-sketching-main/src
# Epoch: 99, Best epoch: 26, Loss: 0.1678, Train: 95.18%, Valid: 76.09%, Test: 72.05%, epoch time: 11.1
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model SEALGCN  
# 0.76 Epoch: 99, Best epoch: 29, Loss: 0.7762, Train: 94.45%, Valid: 76.28%, Test: 72.75%, epoch time: 21.1
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Cora --model SEALGCN > ./output_unihg/output_Cora_SEALGCN.txt 2>&1 &

# Epoch: 99, Best epoch: 36, Loss: 0.2915, Train: 95.62%, Valid: 79.29%, Test: 74.56%, epoch time: 10.9
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset_name Citeseer --model SEALGCN > output_Citeseer_SEALGCN.txt 2>&1 &
# 0.10 Epoch: 99, Best epoch: 43, Loss: 0.3836, Train: 95.58%, Valid: 79.56%, Test: 76.19%, epoch time: 21.0
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Citeseer --model SEALGCN > ./output_unihg/output_Citeseer_SEALGCN.txt 2>&1 &

# Epoch: 99, Best epoch: 95, Loss: 0.2558, Train: 67.81%, Valid: 70.40%, Test: 65.83%, epoch time: 35.6
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset_name Pubmed --model SEALGCN > output_Pubmed_SEALGCN.txt 2>&1 &
# 0.39 Epoch: 99, Best epoch: 98, Loss: 0.4960, Train: 70.34%, Valid: 70.65%, Test: 66.91%, epoch time: 66.4
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Pubmed --model SEALGCN > ./output_unihg/output_Pubmed_SEALGCN.txt 2>&1 &



# session19 BUDDY SEALSAGE Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
conda activate buddy 
cd /gpu-data/qyd_qt/subgraph-sketching-main/src
# Epoch: 99, Best epoch: 36, Loss: 0.1735, Train: 96.51%, Valid: 72.73%, Test: 67.42%, epoch time: 11.9
CUDA_VISIBLE_DEVICES=1 python runners/run.py --dataset_name Cora --model SEALSAGE  
# 0.85 Epoch: 99, Best epoch: 28, Loss: 0.8754, Train: 96.39%, Valid: 72.53%, Test: 70.09%, epoch time: 19.5
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Cora --model SEALSAGE > ./output_unihg/output_Cora_SEALSAGE.txt 2>&1 &

# Epoch: 99, Best epoch: 70, Loss: 0.2161, Train: 94.99%, Valid: 78.20%, Test: 76.14%, epoch time: 10.4
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset_name Citeseer --model SEALSAGE > output_Citeseer_SEALSAGE.txt 2>&1 &
# 0.27 Epoch: 99, Best epoch: 24, Loss: 0.4665, Train: 97.24%, Valid: 78.75%, Test: 76.87%, epoch time: 20.5
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Citeseer --model SEALSAGE > ./output_unihg/output_Citeseer_SEALSAGE.txt 2>&1 &

# Epoch: 99, Best epoch: 83, Loss: 0.2748, Train: 57.92%, Valid: 59.00%, Test: 60.39%, epoch time: 36.0
CUDA_VISIBLE_DEVICES=1 nohup python runners/run.py --dataset_name Pubmed --model SEALSAGE > output_Pubmed_SEALSAGE.txt 2>&1 &
# 0.27 Epoch: 99, Best epoch: 98, Loss: 0.4379, Train: 55.32%, Valid: 60.15%, Test: 61.03%, epoch time: 40.5
CUDA_VISIBLE_DEVICES=4 nohup python runners/run.py --UniHG --dataset_name Pubmed --model SEALSAGE > ./output_unihg/output_Pubmed_SEALSAGE.txt 2>&1 &



# session 20 
# 0.1 TO 0.7
# Run: 01, Epoch: 300, Loss: 0.0022, Acc: 75.42%, precision: 91.28%, recall: 76.01%, f1: 82.94 Time: train_time_epoch: 54.2245192527771, test_time_epoch: 224.6220781803131
CUDA_VISIBLE_DEVICES=5 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.7 > ./lbx/hgd_0.7.txt 2>&1 &
# Run: 01, Epoch: 270, Loss: 0.0020, Acc: 75.74%, precision: 91.11%, recall: 76.27%, f1: 83.03 Time: train_time_epoch: 63.36884808540344, test_time_epoch: 275.3589062690735
CUDA_VISIBLE_DEVICES=5 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.6 > ./lbx/hgd_0.6.txt 2>&1 &
# Run: 01, Epoch: 210, Loss: 0.0017, Acc: 76.04%, precision: 91.07%, recall: 76.76%, f1: 83.31 Time: train_time_epoch: 59.0216109752655, test_time_epoch: 249.12936210632324
CUDA_VISIBLE_DEVICES=5 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.5 > ./lbx/hgd_0.5.txt 2>&1 &
# Run: 01, Epoch: 190, Loss: 0.0014, Acc: 75.72%, precision: 90.96%, recall: 76.43%, f1: 83.06 Time: train_time_epoch: 67.39333033561707, test_time_epoch: 303.643532037735
CUDA_VISIBLE_DEVICES=7 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.4 > ./lbx/hgd_0.4.txt 2>&1 &
# Run: 01, Epoch: 190, Loss: 0.0011, Acc: 75.80%, precision: 90.05%, recall: 76.72%, f1: 82.85 Time: train_time_epoch: 68.39246392250061, test_time_epoch: 291.8633794784546
CUDA_VISIBLE_DEVICES=7 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.3 > ./lbx/hgd_0.3.txt 2>&1 &
# Run: 01, Epoch: 180, Loss: 0.0009, Acc: 75.02%, precision: 88.57%, recall: 76.59%, f1: 82.14 Time: train_time_epoch: 70.69822788238525, test_time_epoch: 303.07331943511963
CUDA_VISIBLE_DEVICES=7 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.2 > ./lbx/hgd_0.2.txt 2>&1 &
# Run: 01, Epoch: 140, Loss: 0.0007, Acc: 75.26%, precision: 90.69%, recall: 75.92%, f1: 82.65 Time: train_time_epoch: 77.41768145561218, test_time_epoch: 283.3235242366791
CUDA_VISIBLE_DEVICES=7 nohup python UniHG-main/models/GNNs/SAGN/sagn.py --use_sagn_embedding --dataset wiki_1M --batch_size 20000 --train_node_dropout 0.1 > ./lbx/hgd_0.1.txt 2>&1 &



# session 21 TGB TASKS hgd

CUDA_VISIBLE_DEVICES=5 python UniHG-main/models/GNNs/SAGN/hgd_st.py > ./wikipedia_hgd_st.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python UniHG-main/models/GNNs/SAGN/hgd_stv2.py > ./wikipedia_hgd_st.txt 2>&1 &

# DyGFormer	0.388 ± 0.006	0.408 ± 0.006
# test ndcg 0.383  val ndcg 0.401
# TGN	0.374 ± 0.001	0.395 ± 0.002	
# DyRep	0.374 ± 0.001	0.394 ± 0.001
/gpu-data/qyd_qt/TGB-main$ 
CUDA_VISIBLE_DEVICES=5 nohup python examples/nodeproppred/tgbn-trade/hgd.py > ./trade_hgd_st.txt 2>&1 &











