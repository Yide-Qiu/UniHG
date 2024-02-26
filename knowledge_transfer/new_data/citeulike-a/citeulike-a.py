import pickle
import pandas as pd

# dataset_path = '/data4/citeulike-a-master/users.dat'
# train_path = './train.txt'
# test_path = './test.txt'

# train_file = open(train_path, 'a')
# test_file = open(test_path, 'a')

# user_id = 0
# with open(dataset_path, 'r') as f:
#     for l in f.readlines():
#         l = l.strip('\n').split(' ')
#         if len(l) > 1:
#             items = [int(i) for i in l]
#             train_items = items[0:int(0.8 * len(items))]
#             test_items = items[int(0.8 * len(items)):]

#             train_line = str(user_id)
#             test_line = str(user_id)

#             for train_item in train_items:
#                 train_line = train_line + ' ' + str(train_item)
#             train_file.write(train_line + '\n')

#             for test_item in test_items:
#                 test_line = test_line + ' ' + str(test_item)
#             test_file.write(test_line + '\n')

#             user_id += 1

# train_file.close()
# test_file.close()

text_path = '/data4/citeulike-a-master/raw-data.csv'
text = pd.read_csv(text_path, encoding='latin1')
item_text = []

for i in range(len(text)):
    item_text.append(str(text['title'][i]))

with open('item_text_citeulike-a.pkl', 'wb') as f:
    pickle.dump(item_text, f)

print(len(item_text))