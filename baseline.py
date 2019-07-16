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
def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl

#====================读入数据====================
train_label= pd.read_csv('训练故障工单.csv',encoding='gbk')
test_label= pd.read_csv('测试故障工单.csv',encoding='gbk')
train_label=train_label.drop_duplicates(['故障发生时间','涉及基站eNBID或小区ECGI',\
                   '故障原因定位（大类）']).reset_index(drop=True) ##去重 （18540，4）


train= pd.read_csv('训练告警.csv',encoding='gbk')
test= pd.read_csv('测试告警.csv',encoding='gbk')
train.columns=['告警开始时间', '告警标题', '涉及基站eNBID或小区ECGI']
test.columns=['告警开始时间', '告警标题', '涉及基站eNBID或小区ECGI']

label=pd.concat([train_label,test_label],axis=0).reset_index(drop=True)
label['故障发生时间']=label['故障发生时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))
label=label.sort_values(by=['涉及基站eNBID或小区ECGI','故障发生时间']).reset_index(drop=True)

data=pd.concat([train,test],axis=0).reset_index(drop=True)
data=data[~data['涉及基站eNBID或小区ECGI'].isnull()]
data=data.drop_duplicates().reset_index(drop=True)
data['告警开始时间']=data['告警开始时间'].apply(lambda x:time.strptime(x,"%Y/%m/%d %H:%M"))


print(train.nunique())
print(test.nunique())

print(train_label.nunique())
print(test_label.nunique())

print('标签分布：',train_label['故障原因定位（大类）'].value_counts())
z=train_label.groupby(['涉及基站eNBID或小区ECGI'],as_index=False)['故障原因定位（大类）'].agg({'n':'nunique'})
print('训练集中，基站出现不同类型故障的个数分布',z['n'].value_counts())

#=================================================================================================
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
假设：故障发生时间 小于等于 告警开始时间（常识好像也是这样，出故障，然后才告警）
下面根据标签数据集，即根据基站的故障发生时间，取提取告警开始时间最近的一条样本（这里可以改）
'''
new_tr=[]
for i,row in train_label.iterrows():
    m=train[(train['涉及基站eNBID或小区ECGI']==row['涉及基站eNBID或小区ECGI'])&\
                (train['告警开始时间']>=row['故障发生时间'])].reset_index(drop=True)
    new_tr.append([m['告警开始时间'][0],m['告警标题'][0]])
new_tr=pd.DataFrame(new_tr,columns=['告警开始时间','告警标题'])
new_te=[]
for i,row in test_label.iterrows():
    m=test[(test['涉及基站eNBID或小区ECGI']==row['涉及基站eNBID或小区ECGI'])&\
                (test['告警开始时间']>=row['故障发生时间'])].reset_index(drop=True)
    new_te.append([m['告警开始时间'][0],m['告警标题'][0]])
new_te=pd.DataFrame(new_te,columns=['告警开始时间','告警标题'])

data=pd.concat([new_tr[['告警标题']],new_te[['告警标题']]],axis=0).reset_index(drop=True)
data1=pd.concat([train_label[['涉及基站eNBID或小区ECGI']],test_label[['涉及基站eNBID或小区ECGI']]],\
                axis=0).reset_index(drop=True)
##对'告警标题' '涉及基站eNBID或小区ECGI' 编码：就两个特征！！
data['f']=one_hot_col(data['告警标题'].astype(str)).transform(data['告警标题'].astype(str))
data['f1']=one_hot_col(data1['涉及基站eNBID或小区ECGI'].astype(str)).transform(data1['涉及基站eNBID或小区ECGI'].astype(str))

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
##标签映射：
train_label['label']=train_label['故障原因定位（大类）'].map(cc)

#==================================================================================================
from sklearn.metrics import roc_auc_score,accuracy_score
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
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
    #skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_xgb=np.zeros((new_train.shape[0],6))
    prediction_xgb=np.zeros((new_test.shape[0],6))
    #for i,(tr,va) in enumerate(skf.split(new_train,y)):
    for i, (tr, va) in enumerate(StratifiedKFold(y, n_folds=5, random_state=42)):
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

###准备训练数据集：
new_train=data[['f','f1']][:len(new_tr)]
new_train['label']=train_label['label']
new_train['涉及基站eNBID或小区ECGI']=train_label['涉及基站eNBID或小区ECGI']
#new_train=new_train[~new_train['涉及基站eNBID或小区ECGI'].isin(final_ids)].reset_index(drop=True) ##去掉只出现一只故障类型的基站，
print(new_train['label'].value_counts())

##准备测试集：
new_test=data[['f','f1']][len(new_tr):].reset_index(drop=True)
new_test['涉及基站eNBID或小区ECGI']=test_label['涉及基站eNBID或小区ECGI']

##模型训练预测：
oof_lgb,prediction_lgb,=\
      xgb_model(np.array(new_train.drop(['涉及基站eNBID或小区ECGI','label'],axis=1)),\
                new_train['label'],\
                np.array(new_test.drop(['涉及基站eNBID或小区ECGI'],axis=1)),0.01)
res=pd.DataFrame(prediction_lgb,columns=['误告警','传输故障','软件故障','电力故障','动环故障','硬件故障'])
res['工单编号']=test_label['工单编号']
res['涉及基站eNBID或小区ECGI']=test_label['涉及基站eNBID或小区ECGI']

##对于只出现过一种故障的基站，测试集中这种故障出现的概率直接为1
for i,row in final.iterrows():
    res.loc[res['涉及基站eNBID或小区ECGI']==row['涉及基站eNBID或小区ECGI'],row['故障原因定位（大类）']]=1

##保存结果并提交：
res[['工单编号', '电力故障', '传输故障', '软件故障', '硬件故障', '动环故障', '误告警'
     ]].to_csv('submit.csv',index=None,encoding='gbk') 
