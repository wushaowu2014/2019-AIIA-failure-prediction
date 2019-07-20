# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:30:33 2019

@author: shaowu
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from collections import Counter
from tqdm import *
def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl

#====================读入数据====================
train_label= pd.read_csv('训练故障工单.csv',encoding='gbk')
test_label= pd.read_csv('测试故障工单.csv',encoding='gbk')
train_label=train_label.drop_duplicates(['故障发生时间','涉及基站eNBID或小区ECGI',\
                   '故障原因定位（大类）']).reset_index(drop=True) ##有些工单一样，去重 （18540，4）


train= pd.read_csv('训练告警.csv',encoding='gbk')
test= pd.read_csv('测试告警.csv',encoding='gbk')
train.columns=['告警开始时间', '告警标题', '涉及基站eNBID或小区ECGI']
test.columns=['告警开始时间', '告警标题', '涉及基站eNBID或小区ECGI']

##训练集中有空行，去掉空行：
train=train[~train['涉及基站eNBID或小区ECGI'].isnull()].reset_index(drop=True) ##有空行

##结合标签：
data_label=pd.concat([train_label,test_label],axis=0).reset_index(drop=True)

##统一时间格式，
data_label['故障发生时间']=data_label['故障发生时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))
##按'涉及基站eNBID或小区ECGI'进行排序，ta再'故障发生时间'进行排序：
#ascending=True，从小到大排序：
data_label=data_label.sort_values(by=['涉及基站eNBID或小区ECGI','故障发生时间'],ascending=True).reset_index(drop=True)

#统一时间格式，方便比较大小：
train['告警开始时间']=train['告警开始时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))
test['告警开始时间']=test['告警开始时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))

train_label['故障发生时间']=train_label['故障发生时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))
test_label['故障发生时间']=test_label['故障发生时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))

##结合标签数据：
data_label=pd.concat([train_label,test_label],axis=0).reset_index(drop=True)
##结合数据集：
data=pd.concat([train,test],axis=0).reset_index(drop=True)
##去重：
data=data.drop_duplicates().reset_index(drop=True)
print('the size  for data_label:',data_label.shape)
print('the size  for data:',data.shape)
#=========================分割线=============================================

print(train.nunique())
print(test.nunique())

print(train_label.nunique())
print(test_label.nunique())

print('*'*50+'\n标签分布：\n',train_label['故障原因定位（大类）'].value_counts())
z=train_label.groupby(['涉及基站eNBID或小区ECGI'],as_index=False)['故障原因定位（大类）'].agg({'n':'nunique'})
print('*'*50+'\n训练集中，基站出现不同类型故障的个数分布\n',z['n'].value_counts())


'''
idea:如果训练集中，基站只出现过一种类型的故障，那么它在测试集中，也只出现这种故障类型；
问题转换：对于基站只出现过一种故障的，以后也是这种故障；对于出现两种以上的基站（最多4种），用模型预测即可。

'''
print(len(set(list(test_label['涉及基站eNBID或小区ECGI'])+list(train_label['涉及基站eNBID或小区ECGI']))))
id_in_tr=list(set(train_label['涉及基站eNBID或小区ECGI'])) ##训练集中的基站
id_in_te=list(set(test_label['涉及基站eNBID或小区ECGI'])) ##测试集中的基站
id_in_tr_te=[i for i in id_in_tr if i in id_in_te] ##同时出现在训练集和测试集的基站
zz=z[z['涉及基站eNBID或小区ECGI'].isin(id_in_tr_te)] ##同时出现在训练集和测试集的基站，它们出现不同类型故障的个数
print(zz['n'].value_counts())

final_ids=list(zz[zz.n==1]['涉及基站eNBID或小区ECGI']) ## 只出现一种类型故障的基站
final=train_label[train_label['涉及基站eNBID或小区ECGI'].isin(final_ids)] ##这部分基站的标签
final=final.drop_duplicates(['涉及基站eNBID或小区ECGI',\
                   '故障原因定位（大类）']).reset_index(drop=True)


#print(test_label['故障发生时间'].min(),test_label['故障发生时间'].max())
#print(train_label['故障发生时间'].min(),train_label['故障发生时间'].max())
#========================================================================================================
'''
假设：故障发生时间 小于等于 告警开始时间（常识好像也是这样，先发生故障，然后才告警）
下面根据标签数据集，即根据基站的故障发生时间，取提取告警开始时间最近的一条样本（这里可以改,改成
提取某段时间的告警标题，或者上一时刻的标签，即历史出现故障类型...）
'''

##下面这个for语句有点慢，可以第一次保存，之后直接读取就可以，
new_data=[]
for i,row in tqdm(data_label.iterrows()):
    m=data[(data['涉及基站eNBID或小区ECGI']==row['涉及基站eNBID或小区ECGI'])&\
                (data['告警开始时间']>=row['故障发生时间'])].reset_index(drop=True)
    new_data.append([m.loc[0,'告警开始时间'],m.loc[0,'告警标题']])
new_data=pd.DataFrame(new_data,columns=['告警开始时间','告警标题'])
#new_data.to_csv('new_data.csv',index=None,encoding='utf8')

data_label=pd.concat([data_label,new_data],axis=1) ##合并数据
data_label['故障发生时间_timestamp']=data_label['故障发生时间'].apply(lambda x:int(time.mktime(x)))
data_label['告警开始时间_timestamp']=data_label['告警开始时间'].apply(lambda x:int(time.mktime(x)))
data_label['告警开始时间_timestamp-故障发生时间_timestamp']=(data_label['告警开始时间_timestamp']-\
                                                          data_label['故障发生时间_timestamp'])/3600

##对'告警标题' '涉及基站eNBID或小区ECGI' 编码：就两个特征！！
data_label['告警标题_code']=one_hot_col(data_label['告警标题'].astype(str)).transform(\
          data_label['告警标题'].astype(str))
data_label['涉及基站eNBID或小区ECGI_code']=one_hot_col(data_label['涉及基站eNBID或小区ECGI'].astype(str)).transform(\
          data_label['涉及基站eNBID或小区ECGI'].astype(str))
'''
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train['conment'] = train['conment'].apply(lambda x:' '.join(jieba.cut(x)))
test['conment'] = test['conment'].apply(lambda x:' '.join(jieba.cut(x)))
#tf-idf特征：
column='conment'
vec = TfidfVectorizer(ngram_range=(1,2),min_df=1, max_df=1.0,use_idf=1,smooth_idf=1, sublinear_tf=1) #这里参数可以改
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
print(trn_term_doc.shape)
'''
#=============================================================================================
def cc(i):
    if i=='误告警':
        return 0
    elif i=='传输故障':
        return 1
    elif i=='软件故障':
        return 2
    elif i=='电力故障':
        return 3
    elif i=='动环故障':
        return 4
    else:
        return 5

#==================================================================================================
from sklearn.metrics import roc_auc_score,accuracy_score
import xgboost as xgb
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
def xgb_model(new_train,y,new_test,lr):
    '''定义模型'''
    xgb_params = {'booster': 'gbtree',
          'eta':lr, 'max_depth': 5, #'subsample': 0.8, 'colsample_bytree': 0.8, 
          #'objective':'binary:logistic',
          'objective': 'multi:softprob',
          #'eval_metric': 'auc',
          #'eval_metric': 'logloss',
           'num_class':6,
          'silent': True,
          }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_xgb=np.zeros((new_train.shape[0],6))
    prediction_xgb=np.zeros((new_test.shape[0],6))
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
    #for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=5, random_state=42)):
        print('fold:',i+1,'training')
        dtrain = xgb.DMatrix(new_train[tr],y[tr])
        dvalid = xgb.DMatrix(new_train[va],y[va])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200, \
        verbose_eval=50, params=xgb_params)
        oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
        prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
    print('the roc_auc_score for train:',accuracy_score(y,np.argmax(oof_xgb,axis=1)))
    prediction_xgb/=5
    return oof_xgb,prediction_xgb

###划分数据集：
new_train=data_label[:train_label.shape[0]] ##训练数据
new_test=data_label[train_label.shape[0]:].reset_index(drop=True)  ##测试数据
new_train['label']=new_train['故障原因定位（大类）'].map(cc) ##标签映射

#new_train=new_train[~new_train['涉及基站eNBID或小区ECGI'].isin(final_ids)].reset_index(drop=True) ##去掉只出现一只故障类型的基站，
print('*'*50+'\n标签分布情况：\n',new_train['label'].value_counts())

##模型训练预测：
oof_lgb,prediction_lgb=\
      xgb_model(np.array(new_train.drop(['工单编号','故障发生时间','涉及基站eNBID或小区ECGI',\
                                         '故障原因定位（大类）','告警开始时间',\
                                         '告警标题','告警开始时间_timestamp',\
                                         '故障发生时间_timestamp','label'],axis=1)),\
                new_train['label'],\
                np.array(new_test.drop(['工单编号','故障发生时间','涉及基站eNBID或小区ECGI',\
                                         '故障原因定位（大类）','告警开始时间',\
                                         '告警标题','告警开始时间_timestamp',\
                                         '故障发生时间_timestamp'],axis=1)),0.01)

res=pd.DataFrame(prediction_lgb,columns=['误告警','传输故障','软件故障','电力故障','动环故障','硬件故障'])
res['工单编号']=new_test['工单编号'].values
res['涉及基站eNBID或小区ECGI']=new_test['涉及基站eNBID或小区ECGI'].values

##对于只出现过一种故障的基站，测试集中这种故障出现的概率直接为1
for i,row in final.iterrows():
    res.loc[res['涉及基站eNBID或小区ECGI']==row['涉及基站eNBID或小区ECGI'],row['故障原因定位（大类）']]=1

##保存结果并提交：
res[['工单编号', '电力故障', '传输故障', '软件故障', '硬件故障', '动环故障', '误告警'
     ]].to_csv('submit.csv',index=None,encoding='gbk') 
