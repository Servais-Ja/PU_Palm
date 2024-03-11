# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
def main():
    feature_array_1 = np.loadtxt('./Feature/allFeature/finalpos.txt', dtype=np.float32)    #基于序列的特征
    feature_array_2 = np.loadtxt('./Feature/allFeature/finalun.txt', dtype=np.float32)

    feature_array_all=np.vstack((feature_array_1,feature_array_2))                      #在竖直方向上堆叠
    label_vector=[]
    path_pos = "./Data/Pos_PSSM"
    path_un = "./Data/Neg_PSSM"
    files_pos = os.listdir(path_pos)                                                    #Get all the file names under the folder 得到指定路径下的文件夹列表
    files_un = os.listdir(path_un)
    for i in range(len(files_pos)):#435 605 969 1492（3066）
        label_vector.append('1')
    for i in range(len(files_un)):#7764
        label_vector.append('-1')
    label_vector = np.array(label_vector,dtype=np.float32)

    #Part 2: data normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    feature_array_all = min_max_scaler.fit_transform(feature_array_all)

    #The independent testing dataset is taken out and cannot participate in model training
    X_train,X_test,y_train,y_test=train_test_split(feature_array_all,label_vector,test_size=0.2,random_state=0,stratify=label_vector)

    c1_list=[0.015625, 0.03123, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    beta_list=[2, 5, 10, 20, 30, 50, 100, 200]

    evaluation_max=0
    c1_max=0
    beta_max=0
    print('Start adjusting parameters...')
    print('*****************************************************')
    evaluation=0
    for c1 in c1_list:
        for beta in beta_list:
            clf = svm.SVC(kernel='linear', probability=True, random_state=42,
                    class_weight={-1: c1 / beta, 1: c1})
            clf = clf.fit(X_train,y_train)
            #Test model using independent testing dataset
            predict_y_test = clf.predict(X_test)
            TP=0
            FN=0
            predict_label1=0
            num_test=len(y_test)
            for i in range(0,len(y_test)):
                if int(y_test[i])==1 and int(predict_y_test[i])==-1:
                    FN=FN+1
                if int(predict_y_test[i])==1:
                    predict_label1=predict_label1+1
                    if int(y_test[i])==1:
                        TP = TP + 1

            Recall=float(TP)/(TP+FN)
            probability_label1=float(predict_label1)/num_test
            evaluation_before=evaluation
            if probability_label1==0:                                                   #解决出现预测全为负的情况
                evaluation=0
            else:
                evaluation=Recall*Recall/probability_label1
            if evaluation_before==1.0 and evaluation==1.0:
                break

            print('c1:%s'%c1)
            print('beta:%s'%beta)
            print('svm Recall:%s'%Recall)
            print('svm predict_label1:%s'%predict_label1)
            print('svm num_test:%s'%num_test)
            print('svm probability_label1:%s'%probability_label1)
            print('svm Evaluation:%s'%evaluation)
            print('*****************************************************')
            if evaluation>evaluation_max:
                evaluation_max=evaluation
                c1_max=c1
                beta_max=beta
                recall_max=Recall
    #输出最优的c1和beta，以及此时的evaluation指标
    print('Parameter Adjustment Results:')
    print('c1_max:%s'%c1_max)
    print('beta_max:%s'%beta_max)
    print('svm Evaluation_max:%s'%evaluation_max)
    print('svm Recall_max:%s'%recall_max)
    with open('调参运行结果.txt','a') as parameterfix_file:
        #parameterfix_file.write(feature_way)                                           #所有POSSUM上基于pssm的特征，单独的效果
        #parameterfix_file.write('\n')

        #parameterfix_file.write('figures choosed:%s\n' % chooselist)                    # 所有效果较好的POSSUM上基于pssm的特征，与学长的三联特征结合的效果
        parameterfix_file.write('Parameter Adjustment Results:\n')
        parameterfix_file.write('c1_max:%s\n' % c1_max)
        parameterfix_file.write('beta_max:%s\n' % beta_max)
        parameterfix_file.write('svm Evaluation_max:%s\n' % evaluation_max)
        parameterfix_file.write('svm Recall_max:%s\n\n' % recall_max)



if __name__=='__main__':
    main()