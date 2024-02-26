
import json
import pdb
import pickle as pkl
from tqdm import tqdm
from utils import entity_select


label_lists = [[1],[1,2,3],[1,3],[2,4,5]]
layered_labels = {2:[5],4:[1,2,5]}
layered_label_lists = []
for e_id in tqdm(range(len(label_lists))):
    layered_label_list = label_lists[e_id]
    for l_id in range(len(label_lists[e_id])):
        if label_lists[e_id][l_id] not in layered_labels.keys():
            continue
        layered_label_list.extend(layered_labels[label_lists[e_id][l_id]])
    layered_label_list = list(set(layered_label_list))
    layered_label_lists.append(layered_label_list)
pdb.set_trace()
# [[1],[1,2,3,5],[1,3],[2,4,5,1]]





# '''
# 小红拿到了一个01串，他准备将若干个字符‘1’染成红色，将若干个字符‘0’染成蓝色，
# 但有个限制，如果一个‘0’和一个‘1’相邻，那么他们不能被同时染色。

# 最多有多少字符可以被染色？

# '''

# def max_colored_characters(s):
#     n = len(s)
#     dp1 = [0] * (n + 1)
#     dp2 = [0] * (n + 1)
#     # pdb.set_trace()
#     for i in range(1, n):
#         if s[i] != s[i-1]:
#             dp1[i] = max(dp1[i-2]+1,dp1[i-1])
#         else:
#             dp1[i] = max(dp1[i-2]+1,dp1[i-1]+1)
            

#     return dp1

# # 示例输入
# s = "0101001"
# n = len(s)
# sel1 = [0] * (n+1)
# sel2 = [0] * (n+1)
# sel2[0] = 1

# for i in range(1, n):
#     if s[i] != s[i-1] and sel1[i-1] == 0:
#         sel1[i] = 1
#     if s[i] == s[i-1]:
#         sel1[i] = 1

# for i in range(1, n):
#     if s[i] != s[i-1] and sel1[i-1] == 0:
#         sel2[i] = 1
#     if s[i] == s[i-1]:
#         sel2[i] = 1

# print(max(sum(sel2),sum(sel1)))




# max_chars = max_colored_characters(s)
# print(max_chars)

# pdb.set_trace()









# def calculate_min_cost(prices, rents):
#     n = len(prices)
#     dp = [float('inf')] * (n + 2)
#     dp[0] = 0

#     for i in range(1, n + 1):
#         dp[i] = min(dp[i], dp[i - 1] + rents[i - 1])
#         dp[i + 1] = min(dp[i + 1], dp[i - 1] + prices[i - 1])

#     return dp[n]

# # 示例输入
# prices = [5, 10, 15]
# rents = [2, 3, 1]

# min_cost = calculate_min_cost(prices, rents)
# print(min_cost)


# pdb.set_trace()




# def calculate_min_cost(prices, rents):
#     n = len(prices)
#     dp = [float('inf')] * (n + 2)
#     dp[0] = 0

#     for i in range(1, n + 1):
#         dp[i] = min(dp[i], dp[i - 1] + rents[i - 1])
#         dp[i + 1] = min(dp[i + 1], dp[i - 1] + prices[i - 1])

#     return dp[n]

# # 示例输入
# prices = [5, 10, 15]
# rents = [2, 3, 1]




# min_cost = calculate_min_cost(prices, rents)
# print(min_cost)


# pdb.set_trace()

















# json_file = 'latest-all.json'
# raw_txt = open('raw_txt.txt', 'r')
# json_all = open(json_file, 'r')

# sum = 0
# num_can_be_js = 0 
# num_have_instance = 0

# for line in raw_txt:
#     # sum += 1
#     # l1 = len(line)
#     # s1 = line
#     # if l1 < 5:
#     #     continue
#     # while s1[-1] != '}' :
#     #     l1 -= 1
#     #     s1 = line[:l1]
#     # js = json.loads(s1)
#     # num_can_be_js += 1
#     # if entity_select(js) == False:
#     #     continue
#     # num_have_instance += 1
#     pdb.set_trace()
#     if num_can_be_js%100 == 0:
#         print(f"js一共{sum}条, 格式正确js共有{num_can_be_js}条, 被选中的js共有{num_have_instance}条。")

# pdb.set_trace()






































# # 可能是utils中对entity的过滤规则有改动，使得运行Adjacent_builder.py时错误选取了部分实体，进而导致邻接阵错误。
# # 如果猜测无误，这次运行第三组数应该是7731--(yes)
# # 等这个跑完，再运行一次Adjacent_builder.py。(not yet.)

# # 然后再运行graph_builder.py上半部分去生成r_edge_index
# # 再生成dgl图, 如果没问题, 应该是7731---

# # 如果不对，那就重新生成。

# # 找到原因，entity2idx错误，为7771----
# # 主要原因是先跑的txt_extractor限制出错
# # 但x_list并没有错 不管如何，改掉entity2idx再去生成Adj是可行的。


