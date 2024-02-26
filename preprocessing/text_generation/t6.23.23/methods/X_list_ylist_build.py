
import pickle as pkl
import pdb
from tqdm import tqdm

X_list_file = 'X_list.pkl'
y_list_file = 'y_list.pkl'

txt = open('raw_txt.txt', 'r', encoding='utf-8')
label_txt = open('label_txt.txt', 'r', encoding='utf-8')

pdb.set_trace()

label = label_txt.readlines()
y = label[0].split(" ")
if '' in y:
    y.remove('')
for i in tqdm(range(len(y))):
    y[i] = [int(y[i])]
print("label list has been done.")

X_list = []
for line in txt:
    X_list.append(line)
print("X list has been done.")

with open(X_list_file, 'wb') as f:
    pkl.dump(X_list, f)
    f.close()

with open(y_list_file, 'wb') as f:
    pkl.dump(y, f)
    f.close()













