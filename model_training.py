import numpy as np
import os
import pickle
from sklearn import svm
from sklearn import preprocessing


feature_array_1 = np.loadtxt('./Feature/allFeature/finalpos.txt', dtype=np.float32)
feature_array_2 = np.loadtxt('./Feature/allFeature/finalun.txt', dtype=np.float32)
feature_array_all = np.vstack((feature_array_1, feature_array_2))  # 在竖直方向上堆叠
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

# Use svm
c1 = 64
beta = 5
clf = svm.SVC(kernel='linear', probability=True, random_state=42,
              class_weight={-1: c1 / beta, 1: c1})
clf = clf.fit(feature_array_all, label_vector)

with open('./model_saved/PU_palm_model.pkl','wb') as f:
    pickle.dump(clf,f)