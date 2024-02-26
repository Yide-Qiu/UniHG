
import pickle as pkl
import pdb
from tqdm import tqdm

label_id2des_dict = pkl.load(open('./../label_id2des_dict.pkl', 'rb'))
new_dict = {} 
for key, value in label_id2des_dict.items():
    if isinstance(value, int):
        new_dict[key] = 'Not Applicable'
    else:
        new_dict[key] = value
label_id2des_dict = new_dict
txt = open('./../raw_txt.txt', 'r', encoding='utf-8')

node_list_file = './../nodes_text_list.pkl'
label_list_file = './../labels_text_list.pkl'

label_list = list(label_id2des_dict.values())
# label = label_txt.readlines()
# y = label[0].split(" ")
# if '' in y:
#     y.remove('')
# for i in tqdm(range(len(y))):
#     y[i] = [int(y[i])]
print("label list has been done.")

node_list = []
for line in txt:
    node_list.append(line)
print("node list has been done.")

with open(node_list_file, 'wb') as f:
    pkl.dump(node_list, f)
    f.close()

with open(label_list_file, 'wb') as f:
    pkl.dump(label_list, f)
    f.close()

pdb.set_trace()











