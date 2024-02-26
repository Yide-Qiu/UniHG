
import json
import time
import pdb
import pickle as pkl
from utils import entity_or_attribute, find_wiki, is_attribute

txt_file = 'latest-all.txt'
relation_file = 'relationship_dict.pkl'

relation_dict = {}
entity_id = 0

with open(txt_file, 'r', encoding='utf-8') as file:
    for line in file:
        line_spl = line.split(" ")
        for spl in line_spl:
            if is_attribute(spl) == False:
                continue
            if spl in relation_dict.keys():
                continue
            else:
                value_spl = ""
                relation_dict[spl] = value_spl
        entity_id += 1
        if entity_id%10 == 0:
            print("已完成{}个实体的关系查询字典映射".format(entity_id))
    file.close()

with open(relation_file, 'wb') as f:
    pkl.dump(relation_dict, f)
    f.close()





