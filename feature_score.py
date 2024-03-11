from sklearn.model_selection import train_test_split
import numpy as np
from trees import PUExtraTrees
import os
from sklearn import preprocessing

array_try_pssm=[[],[]]
seq_Feature_path='./Feature/seqFeature/'
pssm_Feature_path='./Feature/pssmFeature/'

pssmfiles = os.listdir(pssm_Feature_path)  # Get all the file names under the folder 得到指定路径下的文件夹列表
seqfiles = os.listdir(seq_Feature_path)

for file in pssmfiles:
    if 'positive' in file:
        array_try_pssm[0].append(pssm_Feature_path+file)
    elif 'unlabelled' in file:
        array_try_pssm[1].append(pssm_Feature_path + file)
    else:
        print("NameError")
"""
for file in seqfiles:
    if 'positive' in file:
        array_try_pssm[0].append(seq_Feature_path+file)
    elif 'unlabelled' in file:
        array_try_pssm[1].append(seq_Feature_path + file)
    else:
        print("NameError")
"""
feature_array_1 = np.loadtxt(array_try_pssm[0][0], dtype=np.float32)
feature_array_2 = np.loadtxt(array_try_pssm[1][0], dtype=np.float32)
for j in range(3):
    feature_array_choosed_1 = np.loadtxt(array_try_pssm[0][j+1], dtype=np.float32)
    feature_array_choosed_2 = np.loadtxt(array_try_pssm[1][j+1], dtype=np.float32)
    feature_array_1 = np.hstack((feature_array_1, feature_array_choosed_1))
    feature_array_2 = np.hstack((feature_array_2, feature_array_choosed_2))
"""
feature_array_1 = np.loadtxt('./Feature/x_seq_positive.txt', dtype=np.float32)  # 基于序列的特征
feature_array_2 = np.loadtxt('./Feature/x_seq_unlabelled.txt', dtype=np.float32)
"""
# 得到标签向量y
feature_array_all = np.vstack((feature_array_1, feature_array_2))  # 将正样本和未标记样本在竖直方向上堆叠
label_vector = []
path_pos = "./Data/Pos_PSSM"
path_un = "./Data/Neg_PSSM"
files_pos = os.listdir(path_pos)  # Get all the file names under the folder 得到指定路径下的文件夹列表
files_un = os.listdir(path_un)
for i in range(len(files_pos)):
    label_vector.append('1')
for i in range(len(files_un)):
    label_vector.append('-1')
label_vector = np.array(label_vector, dtype=np.float32)
# Part 2: data normalization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
feature_array_all = min_max_scaler.fit_transform(feature_array_all)
x_train, x_test, y_train, y_test = train_test_split(feature_array_all, label_vector, test_size=0.2, random_state=0,
                                                    stratify=label_vector)

#先验知识pi
pi=0.1456
# 将y_train转换为numpy数组
y_train = np.array(y_train)
# 获取正样本和负样本的索引
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == -1)[0]
# 使用索引获取正样本和负样本的特征和标签
x_train_pos = x_train[pos_idx]
x_train_neg = x_train[neg_idx]

n_est=[50,75,100,125,150,200]

max_evaluation=0
for n_estimators in n_est:
    clf = PUExtraTrees(n_estimators = n_estimators,##
                    risk_estimator = 'nnPU',#
                    loss = 'quadratic',#
                    max_depth = 50,##
                    min_samples_leaf = 2,
                    max_features = 'sqrt',#
                    max_candidates = 3,##
                    n_jobs = 4)#


    clf.fit(P=x_train_pos, U=x_train_neg, pi=pi)
    predict_y_test = clf.predict(x_test)
    predict_y_test = predict_y_test[0]
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    predict_label1 = 0
    num_test = len(y_test)
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


    print('svm Recall:%s' % Recall)
    print('svm predict_num:%s' % predict_label1)
    print('svm total_num:%s' % num_test)
    print('svm probability_label1:%s' % probability_label1)
    print('svm Evaluation:%s' % evaluation)


    print('svm Sn:%s' % Sn)
    print('svm Sp:%s' % Sp)
    print('svm ACC:%s' % ACC)
    """
    if max_evaluation<evaluation:
        max_evaluation=evaluation
        importances = clf.feature_importances()
        best_estimator=n_estimators
    """
#np.savetxt("./feature_importances/feature_importances.txt", importances)
#print("best n_estimator:%s"%best_estimator)