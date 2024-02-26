import json
import time
import pdb
import re
import pickle as pkl
from tqdm import tqdm
from utils import entity_or_attribute, label_select, entity_select, is_attribute, is_entity, Bad_Attribute
pdb.set_trace()

# construct the three mapping dictionaries
# json_file = './../../../../rawdata/latest-all.json'
# entity_dict = {}
# label_entity_dict = {}
# relationship_dict = {}
# json_all = open(json_file, 'r')
# num_label_entity = 0
# num_entity = 0
# num_json_line = 0
# num_relationship = 0
# for line in json_all:
#     l1 = len(line)
#     s1 = line
#     if l1 < 5:
#         continue
#     while s1[-1] != '}' :
#         l1 -= 1
#         s1 = line[:l1]
#     js = json.loads(s1)
#     num_json_line += 1

#     if label_select(js) == True:
#         id = js['id']
#         label = js['labels']['en']['value']
#         if id not in label_entity_dict.keys():
#             label_entity_dict[id] = label
#         num_label_entity += 1

#     if entity_select(js) == True:
#         id = js['id']
#         label = js['labels']['en']['value']
#         if id not in entity_dict.keys():
#             entity_dict[id] = label
#         num_entity += 1
        
#         for claim in js['claims'].keys():
#             claim_str = str(claim)
#             if claim_str not in relationship_dict.keys():
#                 relationship_dict[claim_str] = num_relationship
#                 num_relationship += 1
        
#     if num_json_line % 1000 == 0:
#         print("Number of Json Lines: {}".format(num_json_line))
#         print("Number of Label Entities: {}".format(num_label_entity))
#         print("Number of Entities: {}".format(num_entity))
#         print("Number of Relationships: {}".format(num_relationship))

# with open('./../label_entity_dict.pkl', 'wb') as f:
#     pkl.dump(label_entity_dict, f)
#     f.close()
# with open('./../entity_dict.pkl', 'wb') as f:
#     pkl.dump(entity_dict, f)
#     f.close()
# with open('./../relationship_dict.pkl', 'wb') as f:
#     pkl.dump(relationship_dict, f)
#     f.close()
# pdb.set_trace()

entity_id = 0
label_id = 0
edge_id = 0
num_non_eng = 0
json_file = './../../../../rawdata/latest-all.json'
raw_txt = open('./../raw_txt_eng.txt', 'w', encoding='utf-8') 
# label_txt = open('./../label_txt.txt', 'w', encoding='utf-8') # split, arrays, torch.tensor
entity2idx = {}
edge2idx = {}
label_dict = {}
entity_dict = pkl.load(open('./../entity_dict.pkl', 'rb'))
relationship_dict = pkl.load(open('./../relationship_dict2.pkl', 'rb'))
json_all = open(json_file, 'r')
non_english_pattern = re.compile(r'[^\x00-\x7F]+')
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
    entity_txt = label+" be "+description+" . "
    claims_dict = js['claims']
    # instance_of = js['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
    entity2idx[id] = entity_id
    # pdb.set_trace()
    # if instance_of not in label_dict.keys():
    #     label_dict[instance_of] = label_id
    #     label_id += 1
    # label_txt.write(str(label_dict[instance_of])+" ")
    # pdb.set_trace()
    claim_txt = ""
    for claim in js['claims'].keys():
        claim_str = str(claim)
        claim_bef = None
        for claim_idx in range(len(claims_dict[claim_str])):
            if 'datavalue' not in claims_dict[claim_str][claim_idx]['mainsnak']:
                continue
            if claim_str == claim_bef:
                continue
            edge_mainsnak = claims_dict[claim_str][claim_idx]['mainsnak']
            if edge_mainsnak['datatype'] == 'quantity':
                # 对于一些有单位的数值
                amount = edge_mainsnak['datavalue']['value']['amount']
                units = edge_mainsnak['datavalue']['value']['unit']
                unit = units[units.rfind('/')+1:]
                if unit == '1':
                    unit = ''
                claim_txt += claim_str+" "+amount+" "+unit+" . "
                claim_bef = claim_str
            if edge_mainsnak['datatype'] == 'wikibase-item':
                # 对于与其他实体之间的连接
                claim_lat = edge_mainsnak['datavalue']['value']['id']
                claim_txt += claim_str+" "+claim_lat+" . "
    # pdb.set_trace()
    origin_spl = claim_txt.split(" ")
    attribute_spl = claim_txt.split(" ")
    for spl_index in range(len(origin_spl)):
        ss = origin_spl[spl_index]
        if is_attribute(ss):
            if ss not in relationship_dict.keys():
                # print("Error! Property {} is not in dict!".format(ss))
                attribute_spl[spl_index] = ""
                attribute_spl[spl_index+1] = ""
                continue
            # print(ss)
            # print(relationship_dict[ss])
            if Bad_Attribute(relationship_dict[ss]):
                attribute_spl[spl_index] = ""
                attribute_spl[spl_index+1] = ""
                continue
            else:
                if ss not in edge2idx:
                    edge2idx[ss] = edge_id
                    edge_id += 1
                attribute_spl[spl_index] = relationship_dict[ss]
        if is_entity(ss):
            if ss not in entity_dict.keys():
                # print("Error! Entity {} is not in dict!".format(ss))
                attribute_spl[spl_index] = ""
                attribute_spl[spl_index-1] = ""
                continue
            attribute_spl[spl_index] = entity_dict[ss]
    if "" in attribute_spl:
        attribute_spl.remove("")
    line_txt = " ".join(attribute_spl)
    # pdb.set_trace()
    if non_english_pattern.search(entity_txt):
        num_non_eng += 1
        continue
    raw_txt.write(entity_txt+line_txt+"\n")
    entity_id += 1
    if entity_id % 1000 == 0:
        print(f"Has done {entity_id} entities!")
        print(f"Nums of non_eng are {num_non_eng}!")

        raw_txt.flush()
raw_txt.flush()
pdb.set_trace()

# for the Adjacent_builder.py we need to dump entity2idx and edge2idx
# entity2idx : {'Q123:0','Q456:1',...,'Q500:77889900'}
# edge2idx : {'P2429:0','P5126:0',...,'P9294:2082'}

with open('./../edge2idx_dict.pkl', 'wb') as f: 
    pkl.dump(edge2idx, f)
    f.close()



# entity_id = 0
# json_file = './../../../../rawdata/latest-all.json'
# entity2idx = {}
# error_list = []
# json_all = open(json_file, 'r')

# for line in json_all:
#     l1 = len(line)
#     s1 = line
#     if l1 < 5:
#         continue
#     while s1[-1] != '}' :
#         l1 -= 1
#         s1 = line[:l1]
#     js = json.loads(s1)
#     if entity_select(js) == False:
#         continue
#     id = js['id']
#     if id in entity2idx:
#         error_list.append([id,entity2idx[id]])
#     entity2idx[id] = entity_id
#     entity_id += 1
#     if entity_id % 1000 == 0:
#         print(f"Number of recording entities: {entity_id}")

with open('./../entity2idx_dict.pkl', 'wb') as f:
    pkl.dump(entity2idx, f)
    f.close()

pdb.set_trace()

