# -*- coding: utf-8 -*-

from sklearn.metrics import average_precision_score
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def main(pospath,unpath):
    #Import x_positive.txt and x_unlabeled.txt
    #feature_array_1 = np.loadtxt('./Feature/x_positive.txt', dtype=np.float32)
    #feature_array_2 = np.loadtxt('./Feature/x_unlabelled.txt', dtype=np.float32)
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
    feature_array_all= min_max_scaler.fit_transform(feature_array_all)
    
    X_trainset,X_testset,y_trainset,y_testset=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)


    X=X_trainset
    y=y_trainset
    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    skf.get_n_splits(X, y)


    cnt=1
    evaluation_sum=0
    Recall_sum=0
    ap_sum=0
    auc_sum=0
    f1_sum=0
    mcc_sum=0
    sn_sum=0
    sp_sum=0
    acc_sum=0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        c1=64
        beta=5
        clf = svm.SVC(kernel='linear', probability=True, random_state=42, class_weight={-1: c1 / beta, 1: c1})
        clf=clf.fit(X_train,y_train)
        predict_y_test = clf.predict(X_test)
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

        print("第%s次"%cnt)
        print('svm Recall:%s'%Recall)
        print('svm TP:%s'%TP)
        print('svm predict_num:%s'%predict_label1)
        print('svm total_num:%s'%num_test)
        print('svm probability_label1:%s'%probability_label1)
        print('svm Evaluation:%s'%evaluation)

        print('svm AP:%s' % ap)
        print('svm AUC:%s' % roc_auc)
        print('svm F1:%s' % F1)
        print('svm MCC:%s' % MCC)

        print('svm Sn:%s' % Sn)
        print('svm Sp:%s' % Sp)
        print('svm ACC:%s' % ACC)
        print('//////////////////////////////////////////')
        cnt=cnt+1
        evaluation_sum=evaluation_sum+evaluation
        Recall_sum+=Recall
        ap_sum += ap
        auc_sum += roc_auc
        f1_sum += F1
        mcc_sum += MCC
        sn_sum += Sn
        sp_sum += Sp
        acc_sum += ACC
    print("五折交叉检验最终结果：")
    evaluation_average=evaluation_sum/5
    Recall_average=Recall_sum/5
    ap_average=ap_sum/5
    auc_average=auc_sum/5
    f1_average=f1_sum/5
    mcc_average=mcc_sum/5
    sn_average=sn_sum/5
    sp_average=sp_sum/5
    acc_average=acc_sum/5
    print('svm Evaluation:%s'%evaluation_average)
    print('svm Recall:%s' % Recall_average)

    print('svm AP:%s' % ap_average)
    print('svm AUC:%s' % auc_average)
    print('svm F1:%s' % f1_average)
    print('svm MCC:%s' % mcc_average)

    print('svm Sn:%s' % sn_average)
    print('svm Sp:%s' % sp_average)
    print('svm ACC:%s' % acc_average)



    
    
if __name__=='__main__':
    #main('../Feature/allFeature/finalpos.txt','../Feature/allFj eature/finalun.txt')
    seq_filelist=['ct','gtpc_chemf']#'cksaagp','cksaagp_chemf','gaac','gtpc','ct','gtpc_chemf'
    #pssm_filelist=['ab_pssm','dp_pssm','pssm_composition','rpm_pssm','s_fpssm','dpc_pssm','eedp','k_separated_bigrams_pssm']
    for i in seq_filelist:
        main('../Feature/seqFeature/x_'+i+'_positive.txt', '../Feature/seqFeature/x_'+i+'_unlabelled.txt')
    #for i in pssm_filelist:
        #main('../Feature/pssmFeature/x_'+i+'_positive.txt', '../Feature/pssmFeature/x_'+i+'_unlabelled.txt')
    #for i in pssm_filelist:
        #main('../Feature/posFeature/'+i+'_feature.txt', '../Feature/unFeature/'+i+'_feature .txt')






