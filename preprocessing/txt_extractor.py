import json
import time
import pdb
import pickle as pkl
from tqdm import tqdm
from utils import entity_or_attribute, entity_select, is_attribute, is_entity, Bad_Attribute

entity_id = 0
label_id = 0
edge_id = 0
json_file = 'latest-all.json'
raw_txt = open('raw_txt.txt', 'w', encoding='utf-8') 
label_txt = open('label_txt.txt', 'w', encoding='utf-8') # split, arrays, torch.tensor
entity2idx = {}
edge2idx = {}
label_dict = {}
entity_dict = pkl.load(open('entity_dict.pkl', 'rb'))
relationship_dict = pkl.load(open('relationship_dict2.pkl', 'rb'))
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
    entity_txt = label+" be "+description+" . "
    claims_dict = js['claims']
    instance_of = js['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
    entity2idx[id] = entity_id
    # pdb.set_trace()
    if instance_of not in label_dict.keys():
        label_dict[instance_of] = label_id
        label_id += 1
    label_txt.write(str(label_dict[instance_of])+" ")
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
                amount = edge_mainsnak['datavalue']['value']['amount']
                units = edge_mainsnak['datavalue']['value']['unit']
                unit = units[units.rfind('/')+1:]
                if unit == '1':
                    unit = ''
                claim_txt += claim_str+" "+amount+" "+unit+" . "
                claim_bef = claim_str
            if edge_mainsnak['datatype'] == 'wikibase-item':
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
    raw_txt.write(entity_txt+line_txt+"\n")
    entity_id += 1
    if entity_id % 10 == 0:
        print(entity_id)
        raw_txt.flush()
        label_txt.flush()


