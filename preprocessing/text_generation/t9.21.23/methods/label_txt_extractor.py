

import json
import time
import pdb
import torch
import pickle as pkl
from tqdm import tqdm
from utils import entity_or_attribute, label_select, is_attribute, is_entity, Bad_Attribute

'''
we want to extractor id and des. text of label nodes.
'''

entity_id = 0
json_file = './../../../../rawdata/latest-all.json'
# label_entity_dict = pkl.load(open('./../label_entity_dict.pkl', 'rb'))
label_id2idx_dict = pkl.load(open('./../label_id2idx_dict.pkl', 'rb'))
label_txt_list = []

json_all = open(json_file, 'r')
for line in json_all:
    l1 = len(line)
    s1 = line
    if l1 < 5:
        continue
    while s1[-1] != '}' :
        l1 -= 1
        s1 = line[:l1]
    js = json.loads(s1)
    if label_select(js) == False:
        continue
    id = js['id']
    if id not in label_id2idx_dict.keys():
        continue
    label = js['labels']['en']['value']
    description = js['descriptions']['en']['value']
    entity_txt = label+" be "+description+" . "
    label_id2idx_dict[id] = entity_txt
    entity_id += 1
    if entity_id % 1000 == 0:
        print(f"Have collect {entity_id} labels' description.")
print(f"Have collect {entity_id} labels' description.")
with open('./../label_id2des_dict.pkl', 'wb') as f: 
    pkl.dump(label_id2idx_dict, f)
    f.close()




