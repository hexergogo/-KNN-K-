import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

f = pd.read_table('概率.txt',sep=',',header=None).sort_values(by=0)


plt.figure(figsize=(14,18))
plt.style.use('fivethirtyeight')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

#KNN算法K值分析
plt.subplot(3,2,1)
x = f[0]
y = [float(i[:-1])/100  for i in f[1]]
plt.plot(x,y,color='pink')

plt.text(3,y[0],(3,'98.94%'))
plt.text(1299,0.13,(1899,'12.70%'))
plt.title('KNN算法K值分析')
plt.xlabel('K值')
plt.ylabel('准确率')


#k值与准确率关系
plt.subplot(3,2,2)
x = np.array(x.values)
y = np.array(y)

Linear = interp1d(y,x,kind='linear')
y2 = Linear([0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90])
x2 = [98,97,96,95,94,93,92,91,90]
plt.plot(x2,y2,color='r')
plt.bar(x2,y2,width=0.5,align='center',linestyle=':')
for a,b in zip(x2,y2):
    plt.text(a+0.5,b+2,str(b/19)[:3]+'%',ha='center')
plt.xticks([98,97,96,95,94,93,92,91,90])
plt.xlabel('准确率')
plt.ylabel('k值数量')
plt.title('k值与准确率关系')

#错误分析（被判错成何值）
plt.subplot(3,2,3)
f = open('error.txt')
cotent = f.readlines()[0].split(',')
dict = {}
for i in cotent:
    dict[i] = dict.get(i,0) + 1
x = dict.values()
labels = dict.keys()
print(labels)
plt.pie(x,labels=labels,autopct='%.2f%%',shadow=True)
plt.title('错误分析（被判错成何值）')


#错误分析（何值被判错）
plt.subplot(3,2,4)
f = open('error-1.txt')
cotent = f.readlines()[0].split(',')
dict = {}
for i in cotent:
    dict[i] = dict.get(i,0) + 1
x = dict.values()
labels = dict.keys()
print(labels)
plt.pie(x,labels=labels,autopct='%.2f%%',shadow=True)
plt.title('错误分析（何值被判错）')

#测试集分布
plt.subplot(3,2,5)
num_list = [int(i[0]) for i in os.listdir('./digits/testDigits/')]
dict = {}
for i in num_list:
    dict[i] = dict.get(i,0) + 1
x = dict.values()
labels = dict.keys()
plt.pie(x,labels=labels,autopct='%.2f%%',shadow=True)
plt.title('测试集分布')


#训练集分布
plt.subplot(3,2,6)
num_list = [int(i[0]) for i in os.listdir('./digits/trainingDigits/')]
dict = {}
for i in num_list:
    dict[i] = dict.get(i,0) + 1
x = dict.values()
labels = dict.keys()
plt.pie(x,labels=labels,autopct='%.2f%%',shadow=True)
plt.title('训练集分布')

plt.show()