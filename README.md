在做此项目之前，首先要明白何为KNN算法。

一 、 K-近邻算法（KNN）概述
最简单最初级的分类器是将全部的训练数据所对应的类别都记录下来，当测试对象的属性和某个训练对象的属性完全匹配时，便可以对其进行分类。但是怎么可能所有测试对象都会找到与之完全匹配的训练对象呢，其次就是存在一个测试对象同时与多个训练对象匹配，导致一个训练对象被分到了多个类的问题，基于这些问题呢，就产生了KNN。

KNN是通过测量不同特征值之间的距离进行分类。它的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，其中K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

下面通过一个简单的例子说明一下：如下图，绿色圆要被决定赋予哪个类，是红色三角形还是蓝色四方形？如果K=3，由于红色三角形所占比例为2/3，绿色圆将被赋予红色三角形那个类，如果K=5，由于蓝色四方形比例为3/5，因此绿色圆被赋予蓝色四方形类。

在这里插入图片描述

由此也说明了KNN算法的结果很大程度取决于K的选择。

在KNN中，通过计算对象间距离来作为各个对象之间的非相似性指标，避免了对象之间的匹配问题，在这里距离一般使用欧氏距离或曼哈顿距离：
在这里插入图片描述
同时，KNN通过依据k个对象中占优的类别进行决策，而不是单一的对象类别决策。这两点就是KNN算法的优势。

接下来对KNN算法的思想总结一下：就是在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：

1）计算测试数据与各个训练数据之间的距离；

2）按照距离的递增关系进行排序；

3）选取距离最小的K个点；

4）确定前K个点所在类别的出现频率；

5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。

二 .python实现
首先呢，需要说明的是我用的是python3.6，里面有一些用法与2.7还是有些出入。

建立一个KNN.py文件对算法的可行性进行验证，如下：

#coding:utf-8

from numpy import *
import operator

##给出训练数据以及对应的类别
def createDataSet():
    group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels

###通过KNN进行分类
def classify(input,dataSe t,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes
接下来，在命令行窗口输入如下代码：

#-*-coding:utf-8 -*-
import sys
sys.path.append("...文件路径...")
import KNN
from numpy import *
dataSet,labels = KNN.createDataSet()
input = array([1.1,0.3])
K = 3
output = KNN.classify(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)
回车之后的结果为：

测试数据为： [ 1.1 0.3] 分类为： A

答案符合我们的预期，要证明算法的准确性，势必还需要通过处理复杂问题进行验证，之后另行说明。

3、Python实现手写数字识别
训练集和测试集地址如下：
链接：https://pan.baidu.com/s/1lgRDsEp5IGcVn348jjEc8w 提取码：pqm7

测试集和训练集的数字是由1024个01组成
训练集从0-9一共1934个文件
测试集从0-9一共945个文件
在这里插入图片描述

思路：
如果直接以每个文件作为测试集的话，效率低而且相对复杂，所以我的思路是将每个txt中的内容拼成一行，然后将所有的txt都存储在一个txt中。

这样训练集就变成一个1934行，1024列的txt，
同理测试集变成了945行1024列的txt。

每一行代表一个测试集/训练集，每一列代表一个属性，但是只有属性没有类别不行，文件名的第一位就是对应的数字，所以再将属性一起拼接在txt中，所以训练集和测试集就变成了1025列，前1024列为属性，最后一列为类别。

拉直存在一个txt后如下：
在这里插入图片描述

接下来就是代码实现了，为了让程序看起来更python，我使用了类进行封装。

import  pandas as pd #导入pands、numpy并设置别名pd、np
import numpy as np
import os,random

class NumRecog():
    def __init__(self,test,train):
        self.mkfile(test)  ##生成测试集txt
        self.mkfile(train)  ##生成训练集txt

    def mkfile(self,file):  #训练集生成函数
        if os.path.exists(file+'.txt'):  ##如果文件存在则不返回
            self.long = len(os.listdir('./digits/testDigits/')) #获取测试集的长度，以供后面随机选择测试集进行测试
        else:  #否则生成文件
            num_list = os.listdir('./digits/{}/'.format(file)) #获取所有文件
            for i in num_list:
                a = open('./digits/{}/{}'.format(file,i),encoding='utf-8') #打开文件
                with open(r'E:\学习\数据分析\监督学习\{}.txt'.format(file),'a',encoding='utf-8') as f: #写入新文件
                #每个文件撸直并以‘,’隔开每个数字
                 f.write(','.join(list(''.join(a.readlines()).replace('\n','')+i[0]+'\n'))) 
                 
    def DateFrameCreate(self): #数据处理函数
        testF = pd.read_table('testDigits.txt',sep=',') #以，为分隔将文件变成dataframe
        testF.drop(testF.columns[-1],axis=1,inplace=True) #删除最后一列，最后一列为空
        f = pd.read_table('trainingDigits.txt',sep=',') #同理
        f.drop(f.columns[-1],axis=1,inplace=True)
        return self.KNN(testF,f)  #将处理好的数据返回给KNN函数

    def KNN(self,testF,f): #KNN函数
        i = random.randint(1,self.long) #随机取测试集一个测试集
        f_Set = f.iloc[:, :-1]  #最后一列是作为类型，先剔除
        testF_set = testF.iloc[i,:-1] #最后一列是作为类型，先剔除
        testF_set = np.tile(testF_set, [len(f_Set),1]) #把测试集扩展成与训练集同行数
        #生成新的Dataframe
        newF = pd.DataFrame({'数字':f.iloc[:,-1],'相似度':np.sqrt(np.sum((testF_set-f_Set)**2, axis=1))})
        newF = newF.sort_values(by='相似度').head(100) #k=100
        count = newF['数字'].value_counts() #按数字统计
        return count.index[0],testF.iloc[i,-1] #返回统计最多的数字和测试原数字

if __name__ == '__main__':
    numre = NumRecog('testDigits','trainingDigits')
    re = numre.DateFrameCreate()
    print('获取数字：',re[0],'  检测数字为：',re[1])

结果如下：
在这里插入图片描述

4、K值选取对数据准确率的影响
在这里插入图片描述
经过对K = 3~300的不同取值得出：
能得出结论在此事件建中k<总测试集数量10%，准确率在90%以上
在这里插入图片描述
