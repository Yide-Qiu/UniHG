import json
import pandas as pd
import pickle

data_file = open("/data4/yelp/yelp_academic_dataset_review.json")
data = []
for line in data_file:
    data.append(json.loads(line))
checkin_df = pd.DataFrame(data)
data_file.close()

business_file = open('/data4/yelp/yelp_academic_dataset_business.json')
business_with_categories = {}
for line in business_file:
    text = json.loads(line)
    business_with_categories[text['business_id']] = text['categories']
business_file.close()

user_id = []
item_id = []

dataset = {}

for i in range(len(checkin_df)):
    if checkin_df['user_id'][i] == 'unknown':
        print('unknow user id')
        continue
    if checkin_df['business_id'][i] == 'unknown':
        print('unknow item id')
        continue
    date = checkin_df['date'][i]
    _date = date.split('-')[0]
    # if _date == '2018':
    if business_with_categories[checkin_df['business_id'][i]] is None:
        print('存在为none类型的business')
        continue
    if checkin_df['user_id'][i] in dataset:
        dataset[checkin_df['user_id'][i]].append(checkin_df['business_id'][i])
    else:
        dataset[checkin_df['user_id'][i]] = [checkin_df['business_id'][i]]
        # user_id.append(checkin_df['user_id'][i])
        # item_id.append(checkin_df['business_id'][i])

del_keys = []
for key, value in dataset.items():
    if len(value) < 20:
        # del dataset[key]
        del_keys.append(key)

for del_key in del_keys:
    del dataset[del_key]

userID = list(set(dataset.keys()))
itemID = []
for key, value in dataset.items():
    itemID.extend(value)
itemID = list(set(itemID))

user2id = {key:index for index, key in enumerate(userID)}
item2id = {key:index for index, key in enumerate(itemID)}
id2user = {}
# id2item = {}

org_id_user = []
remap_id_user = []
for key, value in user2id.items():
    id2user[value] = key
    org_id_user.append(key)
    remap_id_user.append(value)
user_frame = {'org_id': pd.Series(org_id_user), 'remap_id': pd.Series(remap_id_user)}
user_list = pd.DataFrame(user_frame)

org_id_item = []
remap_id_item = []
business_text = []
for key, value in item2id.items():
    # id2item[value] = key
    org_id_item.append(key)
    remap_id_item.append(value)
    business_text.append(business_with_categories[key])
item_frame = {'org_id': pd.Series(org_id_item), 'remap_id': pd.Series(remap_id_item)}
item_list = pd.DataFrame(item_frame)

with open('./item_text.pkl', 'wb') as f:
    pickle.dump(business_text, f)

user_list.to_csv('user_list.txt', sep=' ', index=0)
item_list.to_csv('item_list.txt', sep=' ', index=0)

dataset_file = open('./dataset.txt', 'a')
train_file = open('./train.txt', 'a')
test_file = open('./test.txt', 'a')

for idx in range(len(id2user)):
    user = id2user[idx]
    line = str(idx)
    train_line = str(idx)
    test_line = str(idx)

    items = dataset[user]
    train_items = items[0:int(0.8 * len(items))]
    test_items = items[int(0.8 * len(items)):]

    for item in items:
        line = line + ' ' + str(item2id[item])
    dataset_file.write(line + '\n')

    for train_item in train_items:
        train_line = train_line + ' ' + str(item2id[train_item])
    train_file.write(train_line + '\n')

    for test_item in test_items:
        test_line = test_line + ' ' + str(item2id[test_item])
    test_file.write(test_line + '\n')
    
dataset_file.close()
train_file.close()
test_file.close()



# data_frame = {'user_id':pd.Series(user_id), 'item_id': pd.Series(item_id)}
# dataset = pd.DataFrame(data_frame)


# '''
# 将u_id喝i_id进行唯一编码
# '''
# userID = list(set(user_id))
# itemID = list(set(item_id))


# yelp = {}

# for i in range(len(checkin_df)):
#     # print(checkin_df['business_id'][i])
#     if checkin_df['user_id'][i][0:2] == '--':
#         yelp[checkin_df['user_id'][i][2:]] = checkin_df['date'][i]
#     else:
#         yelp[checkin_df['user_id'][i]] = checkin_df['date'][i]

# item_list = pd.read_csv('/data4/LightGCN-PyTorch/data/yelp2018/user_list.txt', sep=' ')

# num = 0

# for i in range(len(item_list)):
#     # print(item_list['org_id'][i])
#     if item_list['org_id'][i] in yelp:
#         num += 1
#         print('存在')

# print(f'存在{num}个')