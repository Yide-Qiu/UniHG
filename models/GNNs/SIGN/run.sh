




CUDA_VISIBLE_DEVICES=2 python src/GNNs/SIGN/sign.py --use_sign_embedding --batch_size 200000

CUDA_VISIBLE_DEVICES=1 python src/GNNs/SIGN/sign.py --use_sign_embedding --dataset wiki_1M --batch_size 20000

CUDA_VISIBLE_DEVICES=4 python src/GNNs/SIGN/sign.py --use_sign_embedding --dataset wiki_10M --batch_size 70000


