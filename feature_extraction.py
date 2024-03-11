#This program can extract 1673 features based on five feature extraction methods (FRE,AADP,EEDP, and KSB). 
#The output results are two txt files (x_1673.txt and y.txt) and automatically saved in the current folder.
#The input file is anti_protein_positive_negative.txt
#The output file is used for feature selection in the next step
import re
import numpy as np
import os
from matrixTransformer import *

#以下是基于序列的特征
aa_all='ARNDCQEGHILKMFPSTWYV'
ChemF = ["CM", "AGP", "ILV", "DE","HKR", "FWY", "NQ", "ST"]                 #氨基酸化学特性分组降维
OP_5 = ['G', 'IVFYW', 'ALMEQRK', 'P', 'NDHSTC']                             #氨基酸5组式分组降维

def gaac(seq, gp=aa_all):                                                   #求gaac,输入序列和降维分组(默认不分组)，每个氨基酸出现频率
    aacnt=dict()                                                            #三种分组方式维数分别为20,8,5
    l = len(seq)
    for i in gp:
        aacnt[i]=0
        for j in seq:
            if j in i:
                aacnt[i]=aacnt[i]+1
        aacnt[i]=aacnt[i]/l
    acc_seq=list(aacnt.values())
    return acc_seq

def g_gap(seq, g=0, gp=aa_all):                                             #求G-gap,输入序列,gap和降维分组(默认不分组)，间隔g(默认为0)的两个氨基酸
    gapcnt=dict()                                                           #对特定g的值,三种分组方式维数分别为400,64,25
    l=len(seq)
    for i in gp:
        for j in gp:
            gapcnt[i+j]=0
            for k in range(l-1-g):
                if (seq[k] in i)&(seq[k+1+g] in j):
                    gapcnt[i+j]=gapcnt[i+j]+1
            gapcnt[i + j] = gapcnt[i + j]/(l-1-g)
    gap_seq=list(gapcnt.values())
    return gap_seq

def cksaagp(seq, g=5, gp=OP_5):                                             #求CKSAAGP,输入序列,gap(默认为5)和降维分组(默认OP_5)，小于g的所有g_gap
    cksaagp_seq=[]                                                          #三种分组方式的维数为（400,64,25）*（g+1）
    for i in range(g+1):
        cksaagp_seq=cksaagp_seq+g_gap(seq, i, gp)
    return cksaagp_seq

def gtpc(seq, gp=OP_5):                                                     #求GTPC,输入序列和降维分组(默认OP_5)，相邻三个氨基酸
    tpccnt=dict()                                                           #三种分组方式的维数为8000,512,125
    l=len(seq)
    for i in gp:
        for j in gp:
            for k in gp:
                tpccnt[i+j+k]=0
                for m in range(l-2):
                    if (seq[m] in i)&(seq[m+1] in j)&(seq[m+2] in k):
                        tpccnt[i+j+k]=tpccnt[i+j+k]+1
                tpccnt[i + j + k]=tpccnt[i+j+k]/(l-2)
    tpc_seq=list(tpccnt.values())
    return tpc_seq

def ct(seq):                                                                #一个特殊的tpc,维数为343
    gp=['AGV','ILFP','YMTS','HNQW','RK','DE','C']
    ct_seq=gtpc(seq, gp)
    minf=min(ct_seq)
    maxf=max(ct_seq)
    l=len(ct_seq)
    for i in range(l):
        if maxf==0:
            ct_seq[i]=0
            continue
        ct_seq[i]=(ct_seq[i]-minf)/maxf
    return ct_seq

def fun2(seq_file,seq_name):#cksaagp,cksaagp_chemf,gaac,gtpc
    all_seq=[]
    for line in seq_file:#注意此处有换行符
        if line[0]!='>':
            all_seq.append(line[:-1])
    all_vector=[]
    n=0
    l=len(all_seq)
    for seq in all_seq:
        n=n+1
        if seq_name == 'cksaagp':
            vector=cksaagp(seq)#150
        elif seq_name == 'cksaagp_chemf':
            vector=cksaagp(seq,g=1,gp=ChemF) #128
        elif seq_name == 'gaac':
            vector = gaac(seq) + gaac(seq, gp=ChemF) + gaac(seq, gp=OP_5)  # 33
        elif seq_name == 'ct':
            vector=ct(seq)#343
        elif seq_name == 'gtpc':
            vector=gtpc(seq)#125
        elif seq_name == 'gtpc_chemf':
            vector = gtpc(seq,gp=ChemF)
        all_vector.append(vector)
    return all_vector



#以下是基于PSSM的特征
#此处特征直接使用POSSUM生成的特征矩阵
"""
possum_path='./Feature'
def aac_pssm(path=possum_path):                                             #20
def ab_pssm(path=possum_path):                  #400 1.107 0.895 8 5
def d_fpssm(path=possum_path):                                              #20
def dp_pssm(path=possum_path):                  #240 1.042 0.969 0.5 5      文献找不到
def dpc_pssm(path=possum_path):                                             #400 1.044 0.980 2 5        文献找不到
def edp(path=possum_path):                                                  #20
def eedp(path=possum_path):                                                 #400 1.034 0.977 2 5
def ksb(path=possum_path):                                                  #400 1.061 0.908 1 5
def pse_pssm(path=possum_path):                                             #40
def pssm_ac(path=possum_path):                                              #200
def pssm_cc(path=possum_path):                                              #3800 1.176 0.678 32 5
def pssm_composition(path=possum_path):         #400 1.079 0.947 1 5
def rpm_pssm(path=possum_path):                 #400 1.065 0.954 125 5
def r_pssm(path=possum_path):                                               #110
def s_fpssm(path=possum_path):                  #400 1.030 0.946 125 5
def smoothed_pssm(path=possum_path):                                        #1000 1.105 0.838 4 5
def tpc(path=possum_path):                      #400 该特征会出现报错
def tri_gram_pssm(path=possum_path):                                        #8000
"""
def readToMatrix(input_matrix):
    #print "start to read PSSM matrix"
    PSSM = []
    p = re.compile(r'-*[0-9]+')
    for line, strin in enumerate(input_matrix):
        if line > 2:
            str_vec = []
            overall_vec = strin.split()
            #print len(overall_vec)
            if len(overall_vec) == 0:
                break
            str_vec.extend(overall_vec[1])
            if(len(overall_vec) < 44):
                #print "There is a mistake in the pssm file"
                #print "Try to correct it"
                for cur_str in overall_vec[2:]:
                    str_vec.extend(p.findall(cur_str))
                    if(len(str_vec) >= 21):
                        if(len(str_vec)) >21:
                            #print len(str_vec)
                            #print str_vec
                            #print overall_vec
                            #print "Exit with an error"
                            exit(1)
                        break;
                #print "Done"
            else:
                str_vec = strin.split()[1:42]
            if len(str_vec) == 0:
                break
            #str_vec_positive=map(int, str_vec[1:])
            PSSM.append(str_vec)
    fileinput.close()
    #print "finish to read PSSM matrix"
    PSSM = np.array(PSSM)
    return PSSM

def aac_pssm(input_matrix):
    #print "start aac_pssm function"
    SWITCH = 0
    COUNT = 20
    seq_cn=float(np.shape(input_matrix)[0])
    aac_pssm_matrix=handleRows(input_matrix,SWITCH,COUNT)
    aac_pssm_matrix=np.array(aac_pssm_matrix)
    aac_pssm_vector=average(aac_pssm_matrix,seq_cn)
    #print "end aac_pssm function"
    return aac_pssm_vector

def ab_pssm(input_matrix):
    #print "start ab_pssm function"
    seq_cn=np.shape(input_matrix)[0]
    BLOCK=int(seq_cn/20)
    #print BLOCK
    matrix_final=[]
    for i in range(19):
        tmp=input_matrix[i*BLOCK:(i+1)*BLOCK]
        #print tmp
        matrix_final.append(aac_pssm(tmp)[0])
    tmp=input_matrix[19*BLOCK:]
    #print tmp
    matrix_final.append(aac_pssm(tmp)[0])
    ab_pssm_matrix=average(matrix_final,1.0)
    #print "finish ab_pssm function"
    return ab_pssm_matrix[0]

def dp_pssm(input_matrix):
    #print "start dp_pssm function"
    ALPHA=5
    dp_pssm_matrix=handleMixed2(input_matrix,ALPHA)
    #print "end dp_pssm function"
    return dp_pssm_matrix[0]

def pssm_composition(input_matrix):
    #print "start pssm_composition function"
    SWITCH = 0
    COUNT = 400
    seq_cn=float(np.shape(input_matrix)[0])
    pssm_composition_matrix=handleRows(input_matrix, SWITCH, COUNT)
    pssm_composition_vector=average(pssm_composition_matrix,seq_cn)
    #print "end pssm_composition function"
    return pssm_composition_vector[0]

def rpm_pssm(input_matrix):
    #print "start rpm_pssm function"
    SWITCH = 1
    COUNT = 400
    seq_cn=float(np.shape(input_matrix)[0])
    rpm_pssm_matrix=handleRows(input_matrix,SWITCH,COUNT)
    rpm_pssm_vector=average(rpm_pssm_matrix,seq_cn)
    #print "end rpm_pssm function"
    return rpm_pssm_vector[0]

def s_fpssm(input_matrix):
    #print "start s_fpssm function"
    SWITCH = 2
    COUNT = 400
    seq_cn = 1
    s_fpssm_matrix=handleRows(input_matrix, SWITCH, COUNT)
    s_fpssm_matrix = np.array(s_fpssm_matrix)
    s_fpssm_matrix_shape = np.shape(s_fpssm_matrix)
    matrix_average = [(np.reshape(s_fpssm_matrix, (s_fpssm_matrix_shape[0] * s_fpssm_matrix_shape[1], )))]
    #print "end s_fpssm function"
    return matrix_average[0]

def dpc_pssm(input_matrix):                                                 #A function to get DPC
    STEP = 1
    PART = 0
    ID = 0
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    seq_cn = float(np.shape(input_matrix)[0])                               #序列长度
    dpc_pssm_vector = average(matrix_final, seq_cn-STEP)
    return dpc_pssm_vector[0]

def eedp(input_matrix):                                                     #A function to get EEDP
    STEP = 2
    PART = 0
    ID = 1
    seq_cn = float(np.shape(input_matrix)[0])
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    eedp_vector = average(matrix_final, seq_cn-STEP)
    return eedp_vector[0]                                                   #返回第0行,即该array数组

def k_separated_bigrams_pssm(input_matrix):                                 #A function to get KSB
    STEP = 1
    PART = 1
    ID = 0
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    k_separated_bigrams_pssm_vector=average(matrix_final,10000.0)
    return k_separated_bigrams_pssm_vector[0]

def F1_F2_gram(pssm_array):
    #单个出现频率与氨基酸对出现频率（？21是什么，为什么要有不对应氨基酸名的i）
    #?
    bf={'A':0.082,'R':0.052,'N':0.043,'D':0.058,'C':0.017,'Q':0.039,'E':0.069,
    'G':0.073,'H':0.023,'I':0.057,'L':0.091,'K':0.062,'M':0.018,'F':0.040,
    'P':0.045,'S':0.059,'T':0.054,'W':0.014,'Y':0.035,'V':0.071}
    order={'0':'A','1':'R','2':'N','3':'D','4':'C','5':'Q','6':'E','7':'G',
       '8':'H','9':'I','10':'L','11':'K','12':'M','13':'F','14':'P',
       '15':'S','16':'T','17':'W','18':'Y','19':'V'}
    bf_array = np.array(list(bf.values()),dtype=np.float32)
    M_frequency=np.multiply(pssm_array,bf_array)                            #The exponential part of the frequency matrix is obtained#
                                                                            # 计算矩阵内积    ？乘的bf是啥
    S1_location_array=np.argmax(M_frequency,axis=1)                         #Find out the number position corresponding to the maximum number of each line of the frequency matrix
                                                                            # 获取某一维度最大数的索引，比较行返回列索引的最大值
    S1_location= [str(i) for i in list(S1_location_array)]                  # 将数值变量变为字符变量以匹配order字典
    S1_list=[order[i] if i in order else i for i in S1_location]            #Replace the numeric position with a specific amino acid residue（考虑到可能有别的氨基酸不在这20种里）
    S1=''.join(S1_list)                                                     #Get S1 得到S1序列
    F1_gram=[]
    S1_len=len(S1)
    for n1 in order.values():
        F1_gram.append(S1.count(n1)/S1_len/21)                              #Get V_gram_one 一个20维的矩阵aac‘
    F2_gram=[]
    S1_len=len(S1)-1
    for n1 in order.values():
        for n2 in order.values():
            F2_gram.append(S1.count(n1+n2)*20/S1_len/21)                    #Get V_gram_two 一个400维的矩阵gap’
    return F1_gram + F2_gram



def fun1(pssm_file,pssm_name):                                                        #A function to get AADP（PSSM+DPC）, FRE, EEDP, and KSB
    pssm_array_origin = readToMatrix(pssm_file)
    #The PSSM matrix is obtained
    #Get V_PSSM
    #F_pssm=np.mean(pssm_array,axis=0)                                       #对列求均值（AADP）
    if pssm_name == 'ab_pssm':
        vector=ab_pssm(pssm_array_origin)
    elif pssm_name == 'dp_pssm':
        vector=dp_pssm(pssm_array_origin)
    elif pssm_name == 'pssm_composition':
        vector=pssm_composition(pssm_array_origin)
    elif pssm_name == 'rpm_pssm':
        vector=rpm_pssm(pssm_array_origin)
    elif pssm_name == 's_fpssm':
        vector=s_fpssm(pssm_array_origin)
    elif pssm_name == 'dpc_pssm':
        vector=dpc_pssm(pssm_array_origin)
    elif pssm_name == 'eedp':
        vector=eedp(pssm_array_origin)
    elif pssm_name == 'k_separated_bigrams_pssm':
        vector=k_separated_bigrams_pssm(pssm_array_origin)
    else:
        vector=[]
        print('pssm特征提取方法错误')
    return list(vector)



def main_pssm():
    global aa_all,ChemF,OP_5
    feature_matrix = []
    #Section 2: Feature extraction
    print('Feature extraction starts. Please wait about 15 minutes...')
    #(1) Feature extraction based on fun1 (AADP,FRE,EEDP,KSB)
    print('Extracting AADP,FRE,EEDP,KSB')
    path_p = "./Data/Pos_PSSM"                                              #进入当前目录下的Data
    path_n = "./Data/Neg_PSSM"

    #对正样本提取特征
    files= os.listdir(path_p)                                               #Get all the file names under the folder 得到指定路径下的文件夹列表
    files.sort(key= lambda x:int(x[4:-5]))                                  #Sort by path name 用lamda匿名函数进行排序
    len_files=len(files)
    for file in files:                                                      #对每个PSSM文件（氨基酸序列）单独进行
        #feature_vector=[]
        position = path_p+'/'+ file
        with open(position, "r",encoding='utf-8') as pssm_file:             #省去写close的麻烦
            #feature_vector.extend(fun1(pssm_file))                         #Call fun1 to get features
            feature_matrix.append(fun1(pssm_file))                          #extend将列表中每个元素分别添加进来，append将整个列表作为单个元素添加进来
        print('Sequence '+file[4:-5]+' has finished. ('+str(len_files)+' positive sequences in total)')
    feature_array_all = np.array(feature_matrix,dtype=np.float32)
    #Section 4: Save the result of feature extraction in the current folder
    #The saved format is TXT, and the results can be used for feature selection
    np.savetxt("./Feature/x_positive.txt", feature_array_all)               #Save x

    #对负样本提取特征
    feature_matrix = []
    files = os.listdir(path_n)                                              #Get all the file names under the folder 得到指定路径下的文件夹列表
    files.sort(key=lambda x: int(x[4:-5]))                                  #Sort by path name 用lamda匿名函数进行排序
    len_files = len(files)
    for file in files:
        feature_vector = []
        position = path_n + '/' + file
        with open(position, "r", encoding='utf-8') as pssm_file:            #省去写close的麻烦
            feature_vector.extend(fun1(pssm_file))                          #Call fun1 to get features
            feature_matrix.append(feature_vector)
        print('Sequence ' + file[4:-5] + ' has finished. (' + str(len_files) + ' negtive sequences in total)')
    feature_array_all = np.array(feature_matrix, dtype=np.float32)
    np.savetxt("./Feature/x_unlabelled.txt", feature_array_all)

    print('Completed.')
    print("The results of feature extraction have been saved in the Feature folder.",
          "The file name is x_positive.txt and x_unlabelled.txt")
    
def main_seq(featurename):
    global aa_all, ChemF, OP_5
    pos_file=["./Data/A1B1_cdhit.fa","./Data/A2B2_cdhit.fa","./Data/A3B3_cdhit.fa"]
    neg_file = "./Data/Neg_cdhit.fa"
    #A1_file="./Data/A1B1_cdhit.fa"

    feature_matrix = []
    for file in pos_file:
        with open(file,"r",encoding='utf-8') as Pfile:
            feature_matrix.extend(fun2(Pfile,featurename))
    feature_array_all = np.array(feature_matrix, dtype=np.float32)
    np.savetxt("./Feature/seqFeature/x_cksaagpall_positive.txt", feature_array_all)

    feature_matrix = []
    with open(neg_file,"r",encoding='utf-8') as Nfile:
        feature_matrix.extend(fun2(Nfile,featurename))
    feature_array_all = np.array(feature_matrix, dtype=np.float32)
    np.savetxt("./Feature/seqFeature/x_cksaagpall_unlabelled.txt", feature_array_all)



def main():
    seq_list=['ct','gtpc_chemf']#'cksaagp','cksaagp_chemf','gaac','gtpc','ct','gtpc_chemf'
    pssm_list=['ab_pssm','dp_pssm','pssm_composition','rpm_pssm','s_fpssm','dpc_pssm','eedp','k_separated_bigrams_pssm']

    pos_file = ["./Data/A1B1_cdhit.fa", "./Data/A2B2_cdhit.fa", "./Data/A3B3_cdhit.fa"]
    neg_file = "./Data/Neg_cdhit.fa"
    for feature_name in seq_list:
        print('Feature: '+feature_name)
        feature_matrix = []
        for file in pos_file:
            print('File'+file)
            with open(file, "r", encoding='utf-8') as Pfile:
                feature_matrix.extend(fun2(Pfile,feature_name))
        feature_array_all = np.array(feature_matrix, dtype=np.float32)
        np.savetxt("./Feature/seqFeature/x_"+feature_name+"_positive.txt", feature_array_all)

    for feature_name in seq_list:
        print('Feature: ' + feature_name)
        feature_matrix = []
        with open(neg_file, "r", encoding='utf-8') as Nfile:
            print('File' + neg_file)
            feature_matrix.extend(fun2(Nfile,feature_name))
        feature_array_all = np.array(feature_matrix, dtype=np.float32)
        np.savetxt("./Feature/seqFeature/x_"+feature_name+"_unlabelled.txt", feature_array_all)
    """
    path_p = "./Data/Pos_PSSM"  # 进入当前目录下的Data
    path_n = "./Data/Neg_PSSM"
    files = os.listdir(path_p)  # Get all the file names under the folder 得到指定路径下的文件夹列表
    files.sort(key=lambda x: int(x[4:-5]))  # Sort by path name 用lamda匿名函数进行排序
    len_files = len(files)
    for feature_name in pssm_list:
        print('Feature: '+feature_name)
        feature_matrix = []

        for file in files:  # 对每个PSSM文件（氨基酸序列）单独进行
            pssm_file=fileinput.input(path_p + '/' + file)
            feature_matrix.append(fun1(pssm_file,feature_name))  # extend将列表中每个元素分别添加进来，append将整个列表作为单个元素添加进来
            if int(file[4:-5]) % 500 == 0:
                print('Sequence ' + file[4:-5] + ' has finished. (' + str(len_files) + ' positive sequences in total)')
        feature_array_all = np.array(feature_matrix, dtype=np.float32)
        np.savetxt("./Feature/pssmFeature/x_"+feature_name+"_positive.txt", feature_array_all)  # Save x

    files = os.listdir(path_n)  # Get all the file names under the folder 得到指定路径下的文件夹列表
    files.sort(key=lambda x: int(x[4:-5]))  # Sort by path name 用lamda匿名函数进行排序
    len_files = len(files)
    for feature_name in pssm_list:
        print('Feature: ' + feature_name)
        feature_matrix = []

        for file in files:
            pssm_file = fileinput.input(path_n + '/' + file)
            feature_matrix.append(fun1(pssm_file,feature_name))
            if int(file[4:-5]) % 500 == 0:
                print('Sequence ' + file[4:-5] + ' has finished. (' + str(len_files) + ' negtive sequences in total)')
        feature_array_all = np.array(feature_matrix, dtype=np.float32)
        np.savetxt("./Feature/pssmFeature/x_"+feature_name+"_unlabelled.txt", feature_array_all)
    """


if __name__=='__main__':
    main()