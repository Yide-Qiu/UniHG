import matplotlib.pyplot as plt
# 构建数据
time = [21, 16, 19, 26, 18]
# 中文乱码处理
# plt.rcParams["font.sans-serif"] = ["KaiTi"]
# plt.rcParams['axes.unicode_minus'] = False
# 绘图
plt.bar(range(5),time,width=[0.7,0.7,0.7,0.7,0.7], align = "center",color = "steelblue",alpha = 0.6)

# 添加y轴标签
plt.ylabel("Time(min)")
plt.xlabel("Methods")
# 设置Y轴的刻度范围
plt.ylim([0, 40])
# plt.yticks([0, 40])

# 添加x轴刻度标签
plt.xticks(range(5),['R-MLP','R-SGC','R-SIGN','R-SAGN', 'R-GAMLP'])

# # 添加标题
# plt.title('四个直辖市GDP大比拼')

# 为每个条形图添加数值标签
for x,y in enumerate(time):
    plt.text(x, y + 0.5,'%s' %round(y,1),ha='center')# y+300 标签的坐标

# 显示图形
plt.show()
plt.savefig('figs_time/time_bar.png')
    


