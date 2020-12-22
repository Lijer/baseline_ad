import os
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score,average_precision_score
from util import writeResults_my
p_all = []
r_all = []
f1score_all =[]
roc_auc_all = []
avg_pre_all = []
time_all =[]
data_list = ["apascal", "bank-additional-full_normalised",'lung-1vs5', "probe",'secom',"u2r",'ad','census','creditcard',
             'aid362', 'backdoor', 'celeba', 'chess', 'cmc', 'r10', 'sf', 'w7a']

for t in data_list[-8:]:
    data_path = 'data/{}.csv'.format(t)
    print('*'*10+' lof data : {} '.format(t)+"*"*10)
    with open(data_path,encoding = 'utf-8') as f:
        data = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
    data = data.astype(np.float)
    x = data[:,0:-1]
    y = data[:,-1]
    idx = np.where(y == 0)
    y[idx] = -1

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
        
        start_time = time.time()
        # 计算 LOF 离群因子
        clf = IsolationForest(n_estimators=100)
        clf.fit(X_train)
        y_pred_test = clf.predict(X_test)
        # print(y_test[:15],y_pred_test[:15])
        p_all.append(precision_score(y_test,y_pred_test))
        r_all.append(recall_score(y_test,y_pred_test))
        f1score_all.append(f1_score(y_test,y_pred_test))
        roc_auc_all.append(roc_auc_score(y_test,y_pred_test))
        avg_pre_all.append(average_precision_score(y_test,y_pred_test))
        time_all.append(time.time()-start_time)
        print("precision_score",str(p_all[i-1]))
        print("recall_score",str(r_all[i-1]))
        print("f1_score",str(f1score_all[i-1]))
        print("roc_auc_score",str(roc_auc_all[i-1]))
        print("avg_pre",str(avg_pre_all[i-1]))
        print('耗时:',str(time_all[i-1]))

    print("*"*10+' finished!!! '+"*"*10)
    print("precision_score",str(np.mean(np.array(p_all))), 'std:', str(np.std(p_all)))
    print("recall_score",str(np.mean(np.array(r_all))), 'std:', str(np.std(r_all)))
    print("f1_score",str(np.mean(np.array(f1score_all))), 'std:', str(np.std(f1score_all)))
    print("roc_auc_score",str(np.mean(np.array(roc_auc_all))), 'std:', (np.std(roc_auc_all)))
    print('avg_pre',str(np.mean(np.array(avg_pre_all))), 'std:', str(np.std(avg_pre_all)))
    print('耗时:',str(np.mean(np.array(time_all))), 'std:', str(np.std(time_all)))
    writeResults_my(t, 'if', np.mean(np.array(roc_auc_all)), np.std(roc_auc_all),np.mean(np.array(avg_pre_all)),np.std(avg_pre_all))
    p_all = []
    r_all = []
    f1score_all =[]
    roc_auc_all = []
    avg_pre_all = []
    time_all =[]