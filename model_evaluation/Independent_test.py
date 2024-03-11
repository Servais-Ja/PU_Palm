# -*- coding: utf-8 -*-
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import csv

def main(pospath,unpath,cnt4scorename):
    feature_array_1 = np.loadtxt(pospath, dtype=np.float32)
    feature_array_2 = np.loadtxt(unpath, dtype=np.float32)
    feature_array_all = np.vstack((feature_array_1, feature_array_2))  # 在竖直方向上堆叠
    label_vector = []
    path_pos = "../Data/Pos_PSSM"
    path_un = "../Data/Neg_PSSM"
    files_pos = os.listdir(path_pos)  # Get all the file names under the folder 得到指定路径下的文件夹列表
    files_un = os.listdir(path_un)
    for i in range(len(files_pos)):
        label_vector.append('1')
    for i in range(len(files_un)):
        label_vector.append('-1')
    label_vector = np.array(label_vector, dtype=np.float32)

    #Part 2: data normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    feature_array_all = min_max_scaler.fit_transform(feature_array_all)
    
    #The independent testing dataset is taken out and cannot participate in model training
    X_train,X_test,y_train,y_test=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)
    
    #Use svm
    c1=64
    beta=5
    clf = svm.SVC(kernel='linear', probability=True, random_state=42,
                    class_weight={-1: c1 / beta, 1: c1})
    clf=clf.fit(X_train,y_train)
    #Test model using independent testing dataset
    predict_y_test = clf.predict(X_test)

    scores_y_test = clf.predict_proba(X_test)
    y_test_c=y_test.reshape(-1,1)
    scores_y = np.hstack((scores_y_test,y_test_c))

    TP=0
    FN=0
    TN=0
    FP=0
    predict_label1=0
    num_test=len(y_test)
    for i in range(0, len(y_test)):
        if int(y_test[i]) == 1 and int(predict_y_test[i]) == -1:
            FN = FN + 1
        elif int(y_test[i]) == -1 and int(predict_y_test[i]) == -1:
            TN = TN + 1
        if int(predict_y_test[i]) == 1:
            predict_label1 = predict_label1 + 1
            if int(y_test[i]) == 1:
                TP = TP + 1
            elif int(y_test[i]) == -1:
                FP = FP + 1
    
    #预测为1的概率,保存到csv文件中，用于计算ROC、PR曲线
    csvfilep=open('../SVMscores/scores_p'+cnt4scorename+'.csv','w',newline='')
    csvfileu=open('../SVMscores/scores_u'+cnt4scorename+'.csv','w',newline='')
    writerp = csv.writer(csvfilep)
    writeru = csv.writer(csvfileu)
    for i in scores_y:
        if i[2]==1:
            writerp.writerow([i[1],i[2]])
        else:
            writeru.writerow([i[1],i[2]])
    csvfilep.close()
    csvfileu.close()
    
    Recall=float(TP)/(TP+FN)
    probability_label1=float(predict_label1)/num_test
    evaluation=Recall*Recall/probability_label1

    Sn = float(TP) / (TP + FN)
    Sp = float(TN) / (TN + FP)
    ACC = float((TP + TN)) / (TP + TN + FP + FN)

    prob_predict_y_test = clf.predict_proba(X_test)
    predictions_test = prob_predict_y_test[:, 1]
    y_validation = np.array(y_test, dtype=int)
    fpr, tpr, thresholds = metrics.roc_curve(y_validation, predictions_test, pos_label=1)
    ap = average_precision_score(y_test, predict_y_test)
    roc_auc = auc(fpr, tpr)
    F1 = metrics.f1_score(y_validation, np.array(predict_y_test, int))
    MCC = metrics.matthews_corrcoef(y_validation, np.array(predict_y_test, int))

    print(pospath)
    print('svm Recall:%s'%Recall)
    print('svm predict_num:%s'%predict_label1)
    print('svm total_num:%s'%num_test)
    print('svm probability_label1:%s'%probability_label1)
    print('svm Evaluation:%s'%evaluation)

    print('svm AP:%s' %ap)
    print('svm AUC:%s' %roc_auc)
    print('svm F1:%s' %F1)
    print('svm MCC:%s' %MCC)

    print('svm Sn:%s' %Sn)
    print('svm Sp:%s' %Sp)
    print('svm ACC:%s' %ACC)
    print('***************************************')
    
    stroflist=','.join([str(Recall),str(F1),str(roc_auc),str(ap),str(MCC),str(Sn),str(ACC),str(Sp),str(evaluation)])
    return stroflist+'\n'
    
    
if __name__=='__main__':
    main('../Feature/allFeature/finalpos.txt', '../Feature/allFeature/finalun.txt','final')