
import json
import time
import pdb
import torch
import pickle as pkl
from tqdm import tqdm
from utils import entity_or_attribute, entity_select, is_attribute, is_entity, Bad_Attribute

'''
we want to extractor all 'instance of' of each entity to build a multi-label classifar task.


'''

entity_id = 0
label_id = 0
num_labels = 0
max_length = 0

json_file = './../../../../rawdata/latest-all.json'
label_dict = {}
label_lists = []

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
    num_label = len(js['claims']['P31'])
    label_list = []
    # pdb.set_trace()
    for i in range(num_label):
        # 在2844000断，因此补充：
        if 'datavalue' not in js['claims']['P31'][i]['mainsnak'].keys():
            continue
        l = js['claims']['P31'][i]['mainsnak']['datavalue']['value']['id']
        if l not in label_dict.keys():
            label_dict[l] = label_id
            label_list.append(label_id)
            label_id += 1
            num_labels += 1
        else:
            label_list.append(label_dict[l])
    max_length = max(len(label_list), max_length)
    label_lists.append(label_list)
    # pdb.set_trace()
    entity_id += 1
    if entity_id % 1000 == 0:
        print(f"Have labeled {entity_id} entities.")
        print(f"We now have {num_labels} labels.")

print(f"Have labeled {entity_id} entities.")
print(f"We now have {num_labels} labels.")
print(f"One entity have max {max_length} labels.")

max_length = min(16, max_length)
value = -1
arr_padded = [sublist + [value] * (max_length - len(sublist)) for sublist in label_lists]
arr_truncated = [sublist[:max_length] for sublist in arr_padded]

y = torch.tensor(arr_truncated)

torch.save(y, './../multilabels.pth')
with open('./../label_id2idx_dict.pkl', 'wb') as f:
    pkl.dump(label_dict, f)
    f.close()

pdb.set_trace()

# Here, the multilabels.pth are numbers from 0 to max label.








