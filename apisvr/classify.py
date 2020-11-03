'''use the idea bert encoding and topic model to do semi-classification'''
import codecs
import os
from client.bert_serving.client import BertClient
import numpy as np
import math
import operator

###1、将之前构建的文档-关键词与领域-关键词读出来，返回list/dictionary
#function：read file
def readFile(path):
    word_list = []
    for line in  codecs.open(path,'r',encoding='utf-8',errors='ignore'):
        word_list.append(line.split(':')[0])
    # print(word_list)
    return word_list

#function：读取五大类别的关键词
def readCategoryKeyWord(dir):
    cateDict={}##{每个类别：类别对应的keyword关键词的list[]}

    dir_list=os.listdir(dir)
    for i in range(len(dir_list)):
        path=os.path.join(dir,dir_list[i])
        cate=dir_list[i].replace('.txt','')
        # print(cate)
        cate_keyword_list=readFile(path)
        cateDict[cate]=cate_keyword_list
    # print(cateDict)
    return cateDict

def center(vec):
    c = np.zeros((1, 1024))
    for i in range(len(vec)):
        c += vec[i]
    result=c / len(vec)
    # print(result.shape)
    return result

#function:dot production
def dotProduct(v1, v2):
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    z = v1 * v2.T
    return z

def categorize(category,vec,right_category):
    max_cos=0
    max_cate=''
    for key in category:
        cosine_sim = dotProduct(vec, category[key]) / math.sqrt(dotProduct(vec, vec) * dotProduct(category[key], category[key]))
        if cosine_sim[0][0]>max_cos:
            max_cos=cosine_sim[0][0]
            max_cate=key
    return max_cate

##Euclidean distance
def ouDistance(vector1,vector2):
    ou = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return ou
##cosine distance
def cosDistance(vector1,vector2):
    cosine_sim = dotProduct(vector1, vector2) / math.sqrt(
        dotProduct(vector1, vector1) * dotProduct(vector2, vector2))
    cosine_sim=cosine_sim.A
    return cosine_sim[0][0]

###voting
def categorize_knn(category,vec):

    all_word_cate=[]
    all_word_label=''
    for n in range(len(vec)):
        distDict = {}
        tenCate = []
        for key in category:
            for i in range(len(category[key])):
            # for n in range(len(vec)):
                dist=ouDistance(category[key][i],vec[n])

                distDict[dist] = key

        sortDict = sorted(distDict.items(), key=operator.itemgetter(0), reverse=False)
        sortDict2=dict(sortDict)
        # print('sortDict   ',sortDict2)
        for i, (k, v) in enumerate(sortDict2.items()):
            # print({k: v},'   k,v')
            tenCate.append(v)
            if i == 9:
                tenCate.append(v)
                break
        maxlabel = max(tenCate, key=tenCate.count)
        all_word_cate.append(maxlabel)
    all_word_label=max(all_word_cate, key=all_word_cate.count)
    return all_word_label



###2、bert encoding and classify based on similarity
def classify(test_dir,category_keyword_dict):
    with BertClient(ip='10.0.11.212', port=5555, port_out=5556, show_server_config=True, timeout=100000) as bc:
        ##BBC
        business_vec=bc.encode(category_keyword_dict['business'],is_tokenized=False)
        entertainment_vec = bc.encode(category_keyword_dict['entertainment'], is_tokenized=False)
        politics_vec = bc.encode(category_keyword_dict['politics'], is_tokenized=False)
        sport_vec = bc.encode(category_keyword_dict['sport'], is_tokenized=False)
        tech_vec = bc.encode(category_keyword_dict['tech'], is_tokenized=False)

        ###BBC
        business_c = center(business_vec)
        entertainment_c = center(entertainment_vec)
        politics_c = center(politics_vec)
        sport_c = center(sport_vec)
        tech_c = center(tech_vec)
        ###BBC
        category_dict ={'business':business_c,'entertainment':entertainment_c,'politics':politics_c,'sport':sport_c,'tech':tech_c}
        category_all_vec_dict={'business':business_vec,'entertainment':entertainment_vec,'politics':politics_vec,'sport':sport_vec,'tech':tech_vec}
        test_dir_list = os.listdir(test_dir)
        y_true=[]
        y_pred=[]

        for i in range(len(test_dir_list)):
            right_category = test_dir_list[i]
            second_dir = os.path.join(test_dir, test_dir_list[i])
            second_dir_list = os.listdir(second_dir)
            for j in range(len(second_dir_list)):
                path = os.path.join(second_dir, second_dir_list[j])
                y_true.append(right_category)
                predictFile_keyword_list=readFile(path)
                predict_vec=bc.encode(predictFile_keyword_list,is_tokenized=False)
                predict_c=center(predict_vec)
                predict_category=categorize(category_dict,predict_c,right_category)
                y_pred.append(predict_category)

        print('y_true    ',y_true)
        print('y_pred    ',y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        # t = classification_report(y_true, y_pred, target_names=['business', 'politics', 'tech','entertainment','sport'],digits=4)
        t = classification_report(y_true, y_pred)
        print(t)
dir='output\BBC-train'
category_keyword_dict=readCategoryKeyWord(dir)
test_dir='output\BBC-test'
classify(test_dir,category_keyword_dict)
