
import json
import time
import pdb
from tqdm import tqdm
from utils import entity_or_attribute, find_wiki, entity_select

json_file = 'latest-all.json'
# txt_file = 'latest-all.txt'
# attribute_file = 'attribute-all.txt'
# relation_file = 'relation-all.txt'

# step1: json2txt

# new_entity_flag = 1
# entity_id = 0

# txt = open(txt_file, 'w', encoding='utf-8')
# with open(json_file, 'r') as file:
#     for line in file:
#         # pdb.set_trace()
#         # get this line data of json
#         l1 = len(line)
#         s1 = line[:l1-2]
#         if s1 == None or len(s1) < 5:
#             continue
#         js = json.loads(line[:l1-2])
#         if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
#             continue
#         id = js['id']
#         label = js['labels']['en']['value']
#         description = js['descriptions']['en']['value']
#         entity_txt = label+" be "+description+". "
#         claims_dict = js['claims']
#         claim_txt = ""
#         for claim in js['claims'].keys():
#             claim_str = str(claim)
#             for claim_idx in range(len(claims_dict[claim_str])):
#                 if 'datavalue' not in claims_dict[claim_str][claim_idx]['mainsnak']:
#                     continue
#                 edge_mainsnak = claims_dict[claim_str][claim_idx]['mainsnak']
#                 if edge_mainsnak['datatype'] == 'quantity':
#                     # 对于一些有单位的数值
#                     amount = edge_mainsnak['datavalue']['value']['amount']
#                     units = edge_mainsnak['datavalue']['value']['unit']
#                     unit = units[units.rfind('/')+1:]
#                     if unit == '1':
#                         unit = ''
#                     claim_txt += claim_str+" "+amount+" "+unit+". "
#                 if edge_mainsnak['datatype'] == 'wikibase-item':
#                     # 对于与其他实体之间的连接
#                     claim_lat = edge_mainsnak['datavalue']['value']['id']
#                     claim_txt += claim_str+" "+claim_lat+". "
#         txt.write(entity_txt+claim_txt+"\n")
#         entity_id += 1
#         if entity_id % 10 == 0:
#             print("已完成{}个实体的文本转换".format(entity_id))
#             txt.flush()

# pdb.set_trace()

# step2: txt2attribute

# 属性存在严重的重复问题，一个个去转换计算冗余巨大，设计一个查询映射字典吧。
# 先把关系的代号查出来
# attribute_dict = {}
# relation_dict = {}
# relation = open(relation_file, 'w', encoding='utf-8')
# attribute = open(attribute_file, 'w', encoding='utf-8')
# entity_id = 0
# with open(txt_file, 'r', encoding='utf-8') as file:
#     for line in file:
#         line_spl = line.split(" ")
#         line_txt = line.split(" ")
#         for index in range(len(line_spl)):
#             if entity_or_attribute(line_spl[index]) == False:
#                 continue
#             if line_spl[index] in attribute_dict.keys():
#                 # 如果已被查询过
#                 line_txt[index] = attribute_dict[line_spl[index]]
#             else:
#                 # 如果未查询过
#                 value_apl = find_wiki(line_spl[index])
#                 attribute_dict[line_spl[index]] = value_apl
#                 line_txt[index] = attribute_dict[line_spl[index]]
#         # 将line_index再串起来写入attribute
#         txt = " ".join(line_txt)
#         attribute.write(txt)
#         entity_id += 1
#         if entity_id % 10 == 0:
#             print("已完成{}个实体的代号查询".format(entity_id)) 
#             attribute.flush()


# step3: find label

label_file = 'labels.txt'
label_dict = {}
label_id = 0
entity_id = 0
error_labels = 0
num_no_stance = 0
num_stance = 0


# labels = open(label_file, 'w', encoding='utf-8')
with open(json_file, 'r') as file:
    for line in tqdm(file):
        # pdb.set_trace()
        # get this line data of json
        l1 = len(line)
        s1 = line
        if s1 == None or len(s1) < 5:
            continue
        while s1[-1] != '}' :
            l1 -= 1
            s1 = line[:l1]
        js = json.loads(s1)
        
        if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
            continue
        if 'P31' not in js['claims'].keys():
            # label = js['labels']['en']['value']
            num_no_stance += 1
        else :
            num_stance += 1
#         else:
#             if 'datavalue' not in js['claims']['P31'][0]['mainsnak'].keys():
#                 label = 'lack'
#                 error_labels += 1 
#             else:
#                 label = js['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
#         if label not in label_dict:
#             label_dict[label] = label_id
#             labels.write(str(label_id)+' ')
#             label_id += 1 
#         else:
#             labels.write(str(label_dict[label])+' ')
#         entity_id += 1
#         if entity_id % 10 == 0:
#             print("已完成{}个实体的标签提取".format(entity_id)) 
#             labels.flush()
#         # print(label_dict)
pdb.set_trace()
# # dump label_dict


















