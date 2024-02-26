import torch
import clip
import pdb
import pickle as pkl
from tqdm import tqdm

node_list = pkl.load(open('./../../text_generation/t9.21.23/nodes_text_list.pkl', 'rb'))
label_list = pkl.load(open('./../../text_generation/t9.21.23/labels_text_list.pkl', 'rb'))
node_embedding_file = './../node_embedding.pkl'
label_embedding_file = './../label_embedding.pkl'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# clip+LoRA











