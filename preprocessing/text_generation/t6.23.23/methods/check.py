
import json
import pdb
import pickle as pkl
from utils import entity_select

json_file = 'latest-all.json'
raw_txt = open('raw_txt.txt', 'r')
json_all = open(json_file, 'r')

sum = 0
num_can_be_js = 0 
num_have_instance = 0

for line in raw_txt:
    # sum += 1
    # l1 = len(line)
    # s1 = line
    # if l1 < 5:
    #     continue
    # while s1[-1] != '}' :
    #     l1 -= 1
    #     s1 = line[:l1]
    # js = json.loads(s1)
    # num_can_be_js += 1
    # if entity_select(js) == False:
    #     continue
    # num_have_instance += 1
    pdb.set_trace()
    if num_can_be_js%100 == 0:
        print(f"js一共{sum}条, 格式正确js共有{num_can_be_js}条, 被选中的js共有{num_have_instance}条。")

pdb.set_trace()






































# 可能是utils中对entity的过滤规则有改动，使得运行Adjacent_builder.py时错误选取了部分实体，进而导致邻接阵错误。
# 如果猜测无误，这次运行第三组数应该是7731--(yes)
# 等这个跑完，再运行一次Adjacent_builder.py。(not yet.)

# 然后再运行graph_builder.py上半部分去生成r_edge_index
# 再生成dgl图, 如果没问题, 应该是7731---

# 如果不对，那就重新生成。

# 找到原因，entity2idx错误，为7771----
# 主要原因是先跑的txt_extractor限制出错
# 但x_list并没有错 不管如何，改掉entity2idx再去生成Adj是可行的。


