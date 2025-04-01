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



# session5
|Methods|1|2|3|4|5|6|7|8|9|10|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGC-AFP on 1M|44.76|45.77|43.65|44.98|43.98|44.62|43.71|45.38|44.76|46.36|
|GAMLP-AFP on 1M|47.45|47.57|47.25|48.99|47.25|48.73|49.12|46.68|47.83|48.07|
|SIGN-AFP on 1M|62.03|58.09|62.00|72.27|62.83|62.39|34.04|60.23|70.37|62.55|
|HGD on 1M|75.71|75.96|75.10|75.61|76.03|75.17|75.43|75.74|75.54|76.04|
|SGC-AFP on 10M|63.67|63.29|64.79|64.00|63.28|61.77|-|-|-|-|
|GAMLP-AFP on 10M|59.71|62.35|58.23|61.66|61.78|58.10|-|-|-|-|
|SIGN-AFP on 10M|47.14|78.37|80.15|57.81|63.44|-|-|-|-|-|
|HGD on 10M|88.33|88.13|88.91|88.80|88.69|-|-|-|-|-|



# session5
|Methods|1|2|3|4|5|6|7|8|9|10|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SGC-AFP on 1M|44.76|45.77|43.65|44.98|43.98|44.62|43.71|45.38|44.76|46.36|
|GAMLP-AFP on 1M|47.45|47.57|47.25|48.99|47.25|48.73|49.12|46.68|47.83|48.07|
|SIGN-AFP on 1M|62.03|58.09|62.00|72.27|62.83|62.39|60.23|70.37|62.55|
|SGC on 1M|42.31|45.46|43.02|42.76|41.87|41.87|41.68|43.29|42.76|44.98|
|GAMLP on 1M|44.77|43.83|44.13|46.28|44.46|45.12|46.95|44.58|44.73|44.64|
|SIGN on 1M|55.77|57.67|57.39|62.15|56.09|56.10|52.34|61.78|59.59|



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



# # session15 BUDDY SEALDGCNN Transfer to [ogbl_collab, Cora, Citeseer, Pubmed, PPA, ddi]
# conda activate buddy 
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
CUDA_VISIBLE_DEVICES=5 nohup python examples/nodeproppred/tgbn-trade/hgd.py > ./trade_hgd_st.txt 2>&1 &











