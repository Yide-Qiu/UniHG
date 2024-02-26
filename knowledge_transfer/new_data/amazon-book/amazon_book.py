import json
import pandas as pd
import pickle


data_file = open("/data3/GraphBigModel/extracted_file/meta_Books.json")
check_data = {}
for line in data_file:
    data = json.loads(line)
    check_data[data['asin']] = ','.join(data['category'] + data['description'])
    # data.append(json.loads(line))
# checkin_df = pd.DataFrame(data)
data_file.close()


new_id = 0
new_item_list = {'org_id': [], 'remap_id': [], 'new_id': []}
item_text = []
item_file_path = '/data3/GraphBigModel/amazon-book/item_list.txt'
items = pd.read_csv(item_file_path, delimiter=' ')
for i in range(len(items)):
    if items['org_id'][i] in check_data.keys():
        org_id = items['org_id'][i]

        item_text.append(check_data[org_id])

        new_item_list['org_id'].append(org_id)
        new_item_list['remap_id'].append(items['remap_id'][i])
        new_item_list['new_id'].append(new_id)
        print("true")
        new_id = new_id + 1

new_item_list = pd.DataFrame(new_item_list)
new_item_list.to_csv('new_item_list.txt', sep=' ', index=0)

text_file = open('item_text.pkl', 'wb')
pickle.dump(item_text, text_file)
text_file.close()

print(new_id)


new_item_list = pd.read_csv('new_item_list.txt', delimiter=' ')
train_file_path = '/data3/GraphBigModel/amazon-book/train.txt'
test_file_path = '/data3/GraphBigModel/amazon-book/test.txt'

new_train_file = open('./new_train.txt', 'a')
new_test_file = open('./new_test.txt', 'a')

with open(train_file_path, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').strip(' ').split(' ')
        new_line = line[0]
        if len(line) > 1:
            for i in line[1:]:
                new_item = new_item_list[new_item_list['remap_id'] == int(i)]
                if len(new_item) != 0:
                    new_line = new_line + ' ' + str(new_item['new_id'].item())
        new_train_file.write(new_line + '\n')
new_train_file.close()



with open(test_file_path, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').strip(' ').split(' ')
        new_line = line[0]
        if len(line) > 1:
            for i in line[1:]:
                new_item = new_item_list[new_item_list['remap_id'] == int(i)]
                if len(new_item) != 0:
                    new_line = new_line + ' ' + str(new_item['new_id'].item())    
        new_test_file.write(new_line + '\n')
new_test_file.close()

