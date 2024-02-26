import torch
import clip
import pdb
import pickle as pkl
from tqdm import tqdm
import torch
import torch.nn as nn
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device_ids = [0,1,2,3]

# node_list_file = './../../text_generation/t9.21.23/nodes_text_list.pkl'
# edge_list_file = '../../../NGCF-PyTorch-master/new_data/yelp2018/item_text.pkl'
edge_list_file = '../../../NGCF-PyTorch-master/new_data/amazon-book/item_text.pkl'
node_list = pkl.load(open(edge_list_file, 'rb'))


# amazon_text_path = '/data3/GraphBigModel/NGCF-PyTorch-master/new_data/amazon-book/item_text.pkl'
# amazon_text_list = pkl.load(open(amazon_text_path, 'rb'))


# label_list = pkl.load(open('./../../text_generation/t9.21.23/labels_text_list.pkl', 'rb'))
# node_features_file = './../node_features_frozen.pkl'

# edge_features_file = '../../../NGCF-PyTorch-master/new_data/yelp2018/yelp_item_text_embedding.pkl'
edge_features_file = '../../../NGCF-PyTorch-master/new_data/amazon-book/amazon_book_item_text_embedding.pkl'


# label_features_file = './../label_features_frozen.pkl'
num_nodes = len(node_list)
# num_labels = len(label_list)
node_features = torch.ones(size=(num_nodes,512))
# label_features = torch.ones(size=(num_labels,512))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32")
model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)

shortened_strings = []
num_no_eng = 0
non_english_pattern = re.compile(r'[^\x00-\x7F]+')
max_length = 16
for long_string in tqdm(node_list):
    if len(long_string) <= 75:
        short_string = long_string
    else:
        short_string = long_string[:75]
    match = non_english_pattern.search(short_string)
    if match:
        num_no_eng += 1
        short_string = short_string[:max_length]
    shortened_strings.append(short_string)
node_list = shortened_strings
print(f'raw text has {num_no_eng} no English entities!')

# shortened_strings = []
# for long_string in tqdm(label_list):
#     if len(long_string) <= 75:
#         short_string = long_string
#     else:
#         short_string = long_string[:75]
#     shortened_strings.append(short_string)
# label_list = shortened_strings


# clip
batch_size = 5000
num_node_batches = int(num_nodes/batch_size)+1
for i in tqdm(range(num_node_batches)):
    end = min(num_nodes, (i + 1) * batch_size)
    node_batch = node_list[i * batch_size: end]
    tokens = clip.tokenize(node_batch).cuda()
    with torch.no_grad():
        node_feature = model.module.encode_text(tokens)
    node_features[i*batch_size : end] = node_feature.cpu()

# num_label_batches = int(num_labels/batch_size)+1
# for i in tqdm(range(num_label_batches)):
#     end = min(num_labels, (i + 1) * batch_size)
#     label_batch = label_list[i * batch_size: end]
#     tokens = clip.tokenize(label_batch).cuda()
#     with torch.no_grad():
#         label_feature = model.module.encode_text(tokens)
#     label_features[i*batch_size : end] = label_feature.cpu()

with open(edge_features_file, 'wb') as f:
    pkl.dump(node_features, f)
    f.close()

# with open(label_features_file, 'wb') as f:
#     pkl.dump(label_features, f)
#     f.close()

pdb.set_trace()









