
import json
import time
import pdb
import pickle as pkl
from tqdm import tqdm
from utils import entity_or_attribute, entity_select, is_attribute, is_entity, Bad_Attribute

entity_id = 0
label_id = 0
num_labels = 0
json_file = 'latest-all.json'
# raw_txt = open('raw_txt.txt', 'w', encoding='utf-8') 
# label_txt = open('label_txt.txt', 'w', encoding='utf-8') # split, arrays, torch.tensor
entity_id2des_dict = {}
label_id2idx_dict = {}
label_id2des_dict = {}

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
    if entity_select(js) == False:
        continue
    id = js['id']
    label = js['labels']['en']['value']
    description = js['descriptions']['en']['value']
    entity_txt = label+" be "+description
    if id not in entity_id2des_dict.keys():
        entity_id2des_dict[id] = entity_txt
    # entity_id2des_dict done

    num_label = len(js['claims']['P31'])
    # pdb.set_trace()
    for i in range(num_label):
        # 在2844000断，因此补充：
        if 'datavalue' not in js['claims']['P31'][i]['mainsnak'].keys():
            continue
        l = js['claims']['P31'][i]['mainsnak']['datavalue']['value']['id']
        if l not in label_id2idx_dict.keys():
            label_id2idx_dict[l] = label_id
            label_id += 1
            num_labels += 1
    # label_id2idx_dict done

    entity_id += 1
    if entity_id % 1000 == 0:
        print(f"done {entity_id}")

label_txt_list = []

for id in label_id2idx_dict.keys():
    if id not in entity_id2des_dict.keys():
        print(f"Warning! {id} is not in entity_id2des_dict! May this label id is not in rule!")
        label_txt_list.append("label be empty.")
    else:
        des = entity_id2des_dict[id]
        label_txt_list.append(des)

f1 = open('entity_id2des_dict.pkl', 'wb')
pkl.dump(entity_id2des_dict, f1)
f1.close()

f2 = open('label_id2idx_dict.pkl', 'wb')
pkl.dump(label_id2idx_dict, f2)
f2.close()

f3 = open('label_txt_list.pkl', 'wb')
pkl.dump(label_txt_list, f3)
f3.close()

pdb.set_trace()












