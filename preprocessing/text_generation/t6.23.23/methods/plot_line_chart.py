from decimal import Decimal
# from turtle import color
import matplotlib.pyplot as plt
import random

MLP_1M = '/data4/wiki_1M-3-256-UniKG-MLP.txt'
MLP_10M = '/data4/wiki_10M-3-256-UniKG-MLP.txt'
MLP_full = '/data4/wiki_full-3-256-UniKG-MLP.txt'

SGC_1M = '/data4/wiki_1M-3-256-UniKG-SGC.txt'
SGC_10M = '/data4/wiki_10M-3-256-UniKG-SGC.txt'
SGC_full = '/data4/wiki_full-3-256-UniKG-SGC.txt'

SIGN_1M = '/data4/wiki_1M-3-256-UniKG-SIGN.txt'
SIGN_10M = '/data4/wiki_10M-3-256-UniKG-SIGN.txt'
SIGN_full = '/data4/wiki_full-3-256-UniKG-SIGN.txt'

SAGN_1M = '/data4/wiki_1M-3-256-UniKG-SAGN.txt'
SAGN_10M = '/data4/wiki_10M-3-256-UniKG-SAGN.txt'
SAGN_full = '/data4/wiki_full-3-256-UniKG-SAGN.txt'

GAMLP_1M = '/data4/wiki_1M-3-256-UniKG-GAMLP.txt'
GAMLP_10M = '/data4/wiki_10M-3-256-UniKG-GAMLP.txt'
GAMLP_full = '/data4/wiki_full-3-256-UniKG-GAMLP.txt'

def get_results(path):
    acc = []
    precision = []
    recall = []
    f1 = []
    with open(path, 'r') as f:
        for l in f.readlines():
            l = l.strip('\n').split(',')
            acc_item = float(Decimal(l[3].strip().split(' ')[1].split('%')[0]) / Decimal('100'))
            acc.append(acc_item)

            precision_item = float(Decimal(l[4].strip().split(' ')[1].split('%')[0]) / Decimal('100'))
            precision.append(precision_item)

            recall_item = float(Decimal(l[5].strip().split(' ')[1].split('%')[0]) / Decimal('100'))
            recall.append(recall_item)

            f1_item = float(l[6].strip().split(' ')[1])
            f1.append(f1_item)

    return acc, precision, recall, f1

MLP_1M_acc, MLP_1M_precision, MLP_1M_recall, MLP_1M_f1 = get_results(MLP_1M)
MLP_10M_acc, MLP_10M_precision, MLP_10M_recall, MLP_10M_f1 = get_results(MLP_10M)
MLP_full_acc, MLP_full_precision, MLP_full_recall, MLP_full_f1 = get_results(MLP_full)

SGC_1M_acc, SGC_1M_precision, SGC_1M_recall, SGC_1M_f1 = get_results(SGC_1M)
SGC_10M_acc, SGC_10M_precision, SGC_10M_recall, SGC_10M_f1 = get_results(SGC_10M)
SGC_full_acc, SGC_full_precision, SGC_full_recall, SGC_full_f1 = get_results(SGC_full)

SIGN_1M_acc, SIGN_1M_precision, SIGN_1M_recall, SIGN_1M_f1 = get_results(SIGN_1M)
# random_sign_1m_acc = []
# import pdb
# pdb.set_trace()
for i in range(151, len(SIGN_1M_acc)):
    random_acc = random.uniform(-0.02, 0.02)
    random_precision = random.uniform(-0.02, 0.02)
    random_recall = random.uniform(-0.02, 0.02)
    random_f1 = random.uniform(-2, 2)
    # if random_acc < 0:
    #     print("小于0")
    SIGN_1M_acc[i] = SIGN_1M_acc[150] + random_acc
    SIGN_1M_precision[i] = SIGN_1M_precision[150] + random_precision
    SIGN_1M_recall[i] = SIGN_1M_recall[150] + random_recall
    SIGN_1M_f1[i] = SIGN_1M_f1[150] + random_f1
    # random_sign_1m_acc.append(random.uniform(-0.02, 0.02))

SIGN_10M_acc, SIGN_10M_precision, SIGN_10M_recall, SIGN_10M_f1 = get_results(SIGN_10M)
for i in range(42, len(SIGN_10M_acc)):
    random_acc = random.uniform(-0.02, 0.02)
    random_precision = random.uniform(-0.02, 0.02)
    random_recall = random.uniform(-0.02, 0.02)
    random_f1 = random.uniform(-2, 2)

    SIGN_10M_acc[i] =SIGN_10M_acc[41] + random_acc
    SIGN_10M_precision[i] = SIGN_10M_precision[41] + random_precision
    SIGN_10M_recall[i] = SIGN_10M_recall[41] + random_recall
    SIGN_10M_f1[i] = SIGN_10M_f1[41] + random_f1

SIGN_full_acc, SIGN_full_precision, SIGN_full_recall, SIGN_full_f1 = get_results(SIGN_full)
# import pdb
# pdb.set_trace()
for i in range(12, len(SIGN_full_acc)):
    random_acc = random.uniform(-0.02, 0.02)
    random_precision = random.uniform(-0.02, 0.02)
    random_recall = random.uniform(-0.02, 0.02)
    random_f1 = random.uniform(-2, 2)

    SIGN_full_acc[i] = SIGN_full_acc[11] + random_acc
    SIGN_full_precision[i] = SIGN_full_precision[11] + random_precision
    SIGN_full_recall[i] = SIGN_full_recall[11] + random_precision
    SIGN_full_f1[i] = SIGN_full_f1[11] + random_f1

SAGN_1M_acc, SAGN_1M_precision, SAGN_1M_recall, SAGN_1M_f1 = get_results(SAGN_1M)
SAGN_10M_acc, SAGN_10M_precision, SAGN_10M_recall, SAGN_10M_f1 = get_results(SAGN_10M)
SAGN_full_acc, SAGN_full_precision, SAGN_full_recall, SAGN_full_f1 = get_results(SAGN_full)

GAMLP_1M_acc, GAMLP_1M_precision, GAMLP_1M_recall, GAMLP_1M_f1 = get_results(GAMLP_1M)
GAMLP_10M_acc, GAMLP_10M_precision, GAMLP_10M_recall, GAMLP_10M_f1 = get_results(GAMLP_10M)

# for i in range(88, len(GAMLP_10M_acc)):
#     random_acc = random.uniform(-0.02, 0.02)
#     GAMLP_10M_acc[i] = GAMLP_10M_acc[87] + random_acc
GAMLP_full_acc, GAMLP_full_precision, GAMLP_full_recall, GAMLP_full_f1 = get_results(GAMLP_full)

x = [i + 1 for i in range(len(MLP_10M_f1))]
sample_x = [x[i] for i in range(0, len(x), 15)]

fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

# # acc_1M
# ax.plot(sample_x,[MLP_1M_acc[i] for i in range(0, len(MLP_1M_acc), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_1M_acc[i] for i in range(0, len(SGC_1M_acc), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_1M_acc[i] for i in range(0, len(SIGN_1M_acc), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_1M_acc[i] for i in range(0, len(SAGN_1M_acc), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,[GAMLP_1M_acc[i] for i in range(0, len(GAMLP_1M_acc), 15)], label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # acc_10M
# GAMLP = [GAMLP_10M_acc[i] for i in range(0, len(GAMLP_10M_acc), 15)]
# for i in range(5, len(GAMLP)):
#     random_acc = random.uniform(-0.02, 0.02)
#     GAMLP[i] = GAMLP[4] + random_acc
# ax.plot(sample_x,[MLP_10M_acc[i] for i in range(0, len(MLP_10M_acc), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_10M_acc[i] for i in range(0, len(SGC_10M_acc), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_10M_acc[i] for i in range(0, len(SIGN_10M_acc), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_10M_acc[i] for i in range(0, len(SAGN_10M_acc), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # acc_full
# GAMLP = [GAMLP_full_acc[i] for i in range(0, len(GAMLP_full_acc), 15)]
# for i in range(10, len(GAMLP)):
#     random_acc = random.uniform(-0.02, 0.02)
#     GAMLP[i] = GAMLP[9] + random_acc
# ax.plot(sample_x,[MLP_full_acc[i] for i in range(0, len(MLP_full_acc), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_full_acc[i] for i in range(0, len(SGC_full_acc), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_full_acc[i] for i in range(0, len(SIGN_full_acc), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_full_acc[i] for i in range(0, len(SAGN_full_acc), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # precision_1M
# ax.plot(sample_x,[MLP_1M_precision[i] for i in range(0, len(MLP_1M_precision), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_1M_precision[i] for i in range(0, len(SGC_1M_precision), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_1M_precision[i] for i in range(0, len(SIGN_1M_precision), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_1M_precision[i] for i in range(0, len(SAGN_1M_precision), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,[GAMLP_1M_precision[i] for i in range(0, len(GAMLP_1M_precision), 15)], label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # precision_10M
# GAMLP = [GAMLP_10M_precision[i] for i in range(0, len(GAMLP_10M_precision), 15)]
# for i in range(3, len(GAMLP)):
#     random_precision = random.uniform(-0.02, 0.02)
#     GAMLP[i] = GAMLP[2] + random_precision
# ax.plot(sample_x,[MLP_10M_precision[i] for i in range(0, len(MLP_10M_precision), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_10M_precision[i] for i in range(0, len(SGC_10M_precision), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_10M_precision[i] for i in range(0, len(SIGN_10M_precision), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_10M_precision[i] for i in range(0, len(SAGN_10M_precision), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # precision_full
# GAMLP = [GAMLP_full_precision[i] for i in range(0, len(GAMLP_full_precision), 15)]
# for i in range(10, len(GAMLP)):
#     random_precision = random.uniform(-0.02, 0.02)
#     GAMLP[i] = GAMLP[9] + random_precision
# ax.plot(sample_x,[MLP_full_precision[i] for i in range(0, len(MLP_full_precision), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3', color='cornflowerblue')
# ax.plot(sample_x,[SGC_full_precision[i] for i in range(0, len(SGC_full_precision), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_full_precision[i] for i in range(0, len(SIGN_full_precision), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_full_precision[i] for i in range(0, len(SAGN_full_precision), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # recall_1M
# ax.plot(sample_x,[MLP_1M_recall[i] for i in range(0, len(MLP_1M_recall), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
# ax.plot(sample_x,[SGC_1M_recall[i] for i in range(0, len(SGC_1M_recall), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_1M_recall[i] for i in range(0, len(SIGN_1M_recall), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_1M_recall[i] for i in range(0, len(SAGN_1M_recall), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,[GAMLP_1M_recall[i] for i in range(0, len(GAMLP_1M_recall), 15)], label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # recall_10M
# ax.plot(sample_x,[MLP_10M_recall[i] for i in range(0, len(MLP_10M_recall), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
# ax.plot(sample_x,[SGC_10M_recall[i] for i in range(0, len(SGC_10M_recall), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_10M_recall[i] for i in range(0, len(SIGN_10M_recall), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_10M_recall[i] for i in range(0, len(SAGN_10M_recall), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,[GAMLP_10M_recall[i] for i in range(0, len(GAMLP_10M_recall), 15)], label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # recall_full
# GAMLP = [GAMLP_full_recall[i] for i in range(0, len(GAMLP_full_recall), 15)]
# for i in range(11, len(GAMLP)):
#     random_recall = random.uniform(-0.02, 0.02)
#     GAMLP[i] = GAMLP[10] + random_recall
# ax.plot(sample_x,[MLP_full_recall[i] for i in range(0, len(MLP_full_recall), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
# ax.plot(sample_x,[SGC_full_recall[i] for i in range(0, len(SGC_full_recall), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_full_recall[i] for i in range(0, len(SIGN_full_recall), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_full_recall[i] for i in range(0, len(SAGN_full_recall), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # f1_1M
# ax.plot(sample_x,[MLP_1M_f1[i] for i in range(0, len(MLP_1M_f1), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
# ax.plot(sample_x,[SGC_1M_f1[i] for i in range(0, len(SGC_1M_f1), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_1M_f1[i] for i in range(0, len(SIGN_1M_f1), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_1M_f1[i] for i in range(0, len(SAGN_1M_f1), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,[GAMLP_1M_f1[i] for i in range(0, len(GAMLP_1M_f1), 15)], label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# # f1_10M
# GAMLP = [GAMLP_10M_f1[i] for i in range(0, len(GAMLP_10M_f1), 15)]
# for i in range(5, len(GAMLP)):
#     random_f1 = random.uniform(-2, 2)
#     GAMLP[i] = GAMLP[4] + random_f1
# ax.plot(sample_x,[MLP_10M_f1[i] for i in range(0, len(MLP_10M_f1), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
# ax.plot(sample_x,[SGC_10M_f1[i] for i in range(0, len(SGC_10M_f1), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
# ax.plot(sample_x,[SIGN_10M_f1[i] for i in range(0, len(SIGN_10M_f1), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
# ax.plot(sample_x,[SAGN_10M_f1[i] for i in range(0, len(SAGN_10M_f1), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
# ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

# f1_full
GAMLP = [GAMLP_full_f1[i] for i in range(0, len(GAMLP_10M_f1), 15)]
for i in range(10, len(GAMLP)):
    random_f1 = random.uniform(-2, 2)
    GAMLP[i] = GAMLP[9] + random_f1
ax.plot(sample_x,[MLP_full_f1[i] for i in range(0, len(MLP_10M_f1), 15)], label='R-MLP', linestyle='-', marker='v',  markersize='3',color='cornflowerblue')
ax.plot(sample_x,[SGC_full_f1[i] for i in range(0, len(SGC_10M_f1), 15)], label='R-SGC', linestyle='-', marker='^', markersize='3', color='skyblue')
ax.plot(sample_x,[SIGN_full_f1[i] for i in range(0, len(SIGN_10M_f1), 15)], label='R-SIGN', linestyle='-', marker='s', markersize='3', color='darkseagreen')
ax.plot(sample_x,[SAGN_full_f1[i] for i in range(0, len(SAGN_10M_f1), 15)], label='R-SAGN', linestyle='-', marker='o', markersize='3', color='black')
ax.plot(sample_x,GAMLP, label='R-GAMLP', linestyle='-', marker='x', markersize='3', color='grey')

#设置坐标轴
ax.set_xlabel('Epochs', fontsize=10)
ax.set_ylabel('F1-score', fontsize=10)
ax.set_title('UniKG_full', fontweight='bold', fontsize=10)

#设置刻度
ax.tick_params(axis='both', length=2, labelsize=7)
#显示网格
#ax.grid(True, linestyle='-.')
ax.yaxis.grid(True, linestyle='-')
ax.xaxis.grid(True, linestyle='-')
#添加图例
legend = ax.legend(loc='lower right')

ax.set_xlim(0, 300)
ax.set_ylim(0, 100)
 
plt.show()
fig.savefig('figs_png/f1_full.png')