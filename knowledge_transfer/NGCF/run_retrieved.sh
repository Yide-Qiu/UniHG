
python main.py --dataset citeulike-a --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 220 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 4 --recons True | tee output_citeulike_transferred.txt

python main.py --dataset yelp2018 --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 220 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 4 --recons True | tee output_yelp_transferred.txt

python main.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 220 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 4 --recons True | tee output_amazon_book_transferred.txt
