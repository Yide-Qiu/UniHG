
import pdb
import pickle as pkl
from utils import is_attribute, is_entity, Bad_Attribute

# step2: txt2attribute

# 通过两个映射字典，将latest-all中的代号全部映射为attribute-all中的文本
# warning:注意筛选
# 某些实体仍需要再查(太多就算了)

attribute_file = 'attribute-all.txt'
origin_file = 'latest-all.txt'
entity_need_file = 'entity_need.txt'
relationship_file = 'relationship_dict2.pkl'
entity_file = 'entity_dict.pkl'

attribute_dict = {}
attribute = open(attribute_file, 'w', encoding='utf-8')
entity_need = open(entity_need_file, 'w', encoding='utf-8')
entity_id = 0

relationship_dict = pkl.load(open(relationship_file, 'rb'))
entity_dict = pkl.load(open(entity_file, 'rb'))

with open(origin_file, 'r', encoding='utf-8') as file:
    for line in file:
        origin_spl = line.split(" ")
        attribute_spl = line.split(" ")
        for spl_index in range(len(origin_spl)):
            ss = origin_spl[spl_index]
            if is_attribute(ss):
                if ss not in relationship_dict.keys():
                    print("Error! Property {} is not in dict!".format(ss))
                    entity_need.write(ss+" ")
                    continue
                if Bad_Attribute(relationship_dict[ss]):
                    attribute_spl[spl_index] = ""
                    attribute_spl[spl_index+1] = ""
                else:
                    attribute_spl[spl_index] = relationship_dict[ss]
            if is_entity(ss):
                if ss not in entity_dict.keys():
                    print("Error! Entity {} is not in dict!".format(ss))
                    entity_need.write(ss+" ")
                    continue
                attribute_spl[spl_index] = entity_dict[ss]
        if "" in attribute_spl:
            attribute_spl.remove("")
        line_txt = " ".join(attribute_spl)
        attribute.write(line_txt+"\n")
        entity_id += 1
        if entity_id % 10 == 0:
            print("已完成{}个实体的代号补全".format(entity_id)) 
            attribute.flush()
            entity_need.flush()
pdb.set_trace()
