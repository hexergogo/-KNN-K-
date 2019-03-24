import  pandas as pd #导入pands、numpy并设置别名pd、np
import numpy as np
import os,random

class NumRecog():
    def __init__(self,test,train):
        self.mkfile(test)  ##生成测试集
        self.mkfile(train)  ##生成训练

    def mkfile(self,file):
        if os.path.exists(file+'.txt'):  ##如果文件存在则不返回
            self.long = len(os.listdir('./digits/testDigits/'))
        else:  #否则生成文件
            num_list = os.listdir('./digits/{}/'.format(file)) #获取所有文件
            for i in num_list:
                a = open('./digits/{}/{}'.format(file,i),encoding='utf-8') #打开文件
                with open(r'E:\学习\数据分析\监督学习\{}.txt'.format(file),'a',encoding='utf-8') as f: #写入新文件
                    f.write(','.join(list(''.join(a.readlines()).replace('\n','')+i[0]+'\n'))) #每个文件撸直并以‘,’隔开每个数字

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
        newF = pd.DataFrame({'数字':f.iloc[:,-1],'相似度':np.sqrt(np.sum((testF_set-f_Set)**2, axis=1))}) #生成新的Dataframe
        newF = newF.sort_values(by='相似度').head(900) #k=100
        count = newF['数字'].value_counts() #按数字统计
        return count.index[0],testF.iloc[i,-1] #返回统计最多的数字和测试原数字

if __name__ == '__main__':
    numre = NumRecog('testDigits','trainingDigits')
    re = numre.DateFrameCreate()
    print('获取数字：',re[0],'  检测数字为：',re[1])













 # num,a = 0,0
    # for i in range(945):
    #     try:


#         if int(count.index[0]) == int(testF.iloc[i,-1]):
#             num+=1
#             print(num,count.index[0],testF.iloc[i,-1])
#         a+=1
#         print(a)
#     except:
#         pass
# return num/945