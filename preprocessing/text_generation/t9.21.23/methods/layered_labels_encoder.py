
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

layered_labels = pkl.load(open('./../layered_labels.pkl', 'rb'))
layered_label_lists = []
max_length = 0
for e_id in tqdm(range(len(label_lists))):
    layered_label_list = label_lists[e_id]
    for l_id in range(len(label_lists[e_id])):
        if label_lists[e_id][l_id] not in layered_labels.keys():
            continue
        layered_label_list.extend(layered_labels[label_lists[e_id][l_id]])
    layered_label_list = list(set(layered_label_list))
    layered_label_lists.append(layered_label_list)
    max_length = max(max_length, len(layered_label_list))

pdb.set_trace()
num_saved_labels = 64
max_length = min(num_saved_labels, max_length)
value = -1
arr_padded = [sublist + [value] * (max_length - len(sublist)) for sublist in layered_label_lists]
arr_truncated = [sublist[:max_length] for sublist in arr_padded]

y = torch.tensor(arr_truncated)

torch.save(y, './../layered_label.pth')
pdb.set_trace()








