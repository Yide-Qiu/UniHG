
import json
import pdb
import pickle as pkl
from utils import entity_select, is_attribute, is_entity, Bad_Attribute

entity2idx_file = 'entity2idx2.pkl'
edge2idx_file = 'edge2idx.pkl'
relationship_file = 'relationship_dict2.pkl'
json_file = 'latest-all.json'
adjacent = open('relative_adjacent3.txt' , 'w', encoding='utf-8')
entity_id = 0
num_edges = 0

relationship_dict = pkl.load(open(relationship_file, 'rb'))
entity2idx_dict = pkl.load(open(entity2idx_file, 'rb'))
relationship2idx_dict = pkl.load(open(edge2idx_file, 'rb'))

# pdb.set_trace()

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
    claims_dict = js['claims']
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
    for spl_index in range(len(origin_spl)):
        ss = origin_spl[spl_index]
        if is_attribute(ss):
            if ss not in relationship_dict.keys():
                continue
            if Bad_Attribute(relationship_dict[ss]):
                continue
            else:
                if is_entity(origin_spl[spl_index+1]) and origin_spl[spl_index+1] in entity2idx_dict.keys():
                    num_edges += 1
                    adjacent.write(str(entity_id)+" "+str(relationship2idx_dict[ss])+" "+str(entity2idx_dict[origin_spl[spl_index+1]])+"\n")
    adjacent.write(str(entity_id)+" "+str(2082)+" "+str(entity_id)+"\n")
    entity_id += 1
    if entity_id % 100 == 0:
        print("邻接矩阵已读入{}个实体".format(entity_id))
        print("当前邻接矩阵包含边数为{}".format(num_edges))
        adjacent.flush()

adjacent.flush()
pdb.set_trace()











