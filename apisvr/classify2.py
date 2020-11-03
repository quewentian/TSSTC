'''用bertencoding分类'''
import codecs
import os
from client.bert_serving.client import BertClient
import numpy as np
import math
import operator

###1、将之前构建的文档-关键词与领域-关键词读出来，返回list/dictionary
#function：读取每篇文档关键词
def readFile(path):
    word_list = []
    for line in  codecs.open(path,'r',encoding='utf-8',errors='ignore'):
        # print(type(line))
        # word_list=[]
        # print(line.split(':'))
        word_list.append(line.split(':')[0])
    # print(word_list)
    return word_list

#function：读取五大类别的关键词
def readCategoryKeyWord(dir):
    cateDict={}##{每个类别：类别对应的keyword关键词的list[]}

    dir_list=os.listdir(dir)
    # print(dir_list)
    for i in range(len(dir_list)):
        path=os.path.join(dir,dir_list[i])
        cate=dir_list[i].replace('.txt','')
        # print(cate)
        cate_keyword_list=readFile(path)
        cateDict[cate]=cate_keyword_list
    # print(cateDict)
    return cateDict

def center(vec):###球心
    # print(vec.shape)
    c = np.zeros((1, 1024))
    for i in range(len(vec)):
        c += vec[i]
    result=c / len(vec)
    # print(result.shape)
    return result

def dotProduct(v1, v2):###点乘
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    z = v1 * v2.T
    return z

def categorize(category,vec,right_category):###分类
    max_cos=0   #最大的相似度
    max_cate='' #最大相似度的类
    for key in category:
        # print(category[key])
        # print(category[key].shape)
        # print(vec,'  vec')
        # print(vec.shape,'   vec shape')

        # print(type(vec))
        # print(category[key],'   category[key]')
        # print(type(category[key]),'   type category[key]')
        cosine_sim = dotProduct(vec, category[key]) / math.sqrt(dotProduct(vec, vec) * dotProduct(category[key], category[key]))
        # print('每类得分   ',key,'：   ',cosine_sim[0][0])
        if cosine_sim[0][0]>max_cos:
            max_cos=cosine_sim[0][0]
            max_cate=key
    print('最可能的类别是:    ',max_cate,'    真实的标签是：   ',right_category)
    return max_cate

##计算欧式距离
def ouDistance(vector1,vector2):
    ou = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return ou
##计算COS距离
def cosDistance(vector1,vector2):
    # cos = np.sqrt(np.sum(np.square(vector1 - vector2)))
    cosine_sim = dotProduct(vector1, vector2) / math.sqrt(
        dotProduct(vector1, vector1) * dotProduct(vector2, vector2))
    cosine_sim=cosine_sim.A
    return cosine_sim[0][0]

###实在不行搞个投票机制
def categorize_knn(category,vec):
    # distDict={}###距离词典{‘距离大小’，类别}，待会按照距离大小排序，最后取前十的items
    # tenCate=[]
    all_word_cate=[]
    all_word_label=''
    for n in range(len(vec)):########每个词10个票。（之后挑出1个最可能的类）
        distDict = {}  ###距离词典{‘距离大小’，类别}，待会按照距离大小排序，最后取前十的items
        tenCate = []
        for key in category:
            for i in range(len(category[key])):
            # for n in range(len(vec)):
                dist=ouDistance(category[key][i],vec[n])
                # dist =cosDistance(category[key][i],vec[n])
                # distDict[str(dist)]=key
                # print(dist,'   dist2333    ',  type(dist))
                # print(dist,'   dist[0][0]2333    ',type(dist[0][0]))
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
        # print('十类：  !!! ',tenCate,'   222')
    # for j in range(len(tenCate)):
        maxlabel = max(tenCate, key=tenCate.count)#######取出重复出现次数最多的类别
        # print('最可能的类别是   ',maxlabel)
        all_word_cate.append(maxlabel)
        # print('一个句子的词对应的所有最可能类别   ',all_word_cate)
    # print('all_word_cate   ',all_word_cate)
    all_word_label=max(all_word_cate, key=all_word_cate.count)
    # print('最可能的类别是   ',all_word_label)
    return all_word_label



###2、将文档与领域的关键词，用bert编码后，分别作相似度匹配
def classify(test_dir,category_keyword_dict):
    with BertClient(ip='10.0.11.212', port=5555, port_out=5556, show_server_config=True, timeout=100000000) as bc:
        ##BBC
        # business_vec=bc.encode(category_keyword_dict['business'],is_tokenized=False)
        # entertainment_vec = bc.encode(category_keyword_dict['entertainment'], is_tokenized=False)
        # politics_vec = bc.encode(category_keyword_dict['politics'], is_tokenized=False)
        # sport_vec = bc.encode(category_keyword_dict['sport'], is_tokenized=False)
        # tech_vec = bc.encode(category_keyword_dict['tech'], is_tokenized=False)

        ##4——类
        # accounts_vec = bc.encode(category_keyword_dict['accounts'], is_tokenized=False)
        # biology_vec = bc.encode(category_keyword_dict['biology'], is_tokenized=False)
        # geography_vec = bc.encode(category_keyword_dict['geography'], is_tokenized=False)
        # physics_vec = bc.encode(category_keyword_dict['physics'], is_tokenized=False)

        ###5 abstract
        Business_vec = bc.encode(category_keyword_dict['Business'], is_tokenized=False)
        CSAI_vec = bc.encode(category_keyword_dict['CSAI'], is_tokenized=False)
        Law_vec = bc.encode(category_keyword_dict['Law'], is_tokenized=False)
        Sociology_vec = bc.encode(category_keyword_dict['Sociology'], is_tokenized=False)
        Trans_vec = bc.encode(category_keyword_dict['Trans'], is_tokenized=False)

        # print(business_vec.shape,'   111')###(600, 1024)
        # print(entertainment_vec.shape, '   112')
        # print(politics_vec.shape, '   113')
        # print(sport_vec.shape, '   114')
        # print(tech_vec.shape, '   115')

        ###BBC
        # business_c = center(business_vec)
        # entertainment_c = center(entertainment_vec)
        # politics_c = center(politics_vec)
        # sport_c = center(sport_vec)
        # tech_c = center(tech_vec)

        ##4类
        # accounts_c=center(accounts_vec)
        # biology_c=center(biology_vec)
        # geography_c=center(geography_vec)
        # physics_c=center(physics_vec)

        ###5 abstract
        Business_c = center(Business_vec)
        CSAI_c = center(CSAI_vec)
        Law_c = center(Law_vec)
        Sociology_c = center(Sociology_vec)
        Trans_c = center(Trans_vec)

        ###BBC
        # category_dict ={'business':business_c,'entertainment':entertainment_c,'politics':politics_c,'sport':sport_c,'tech':tech_c}
        # category_all_vec_dict={'business':business_vec,'entertainment':entertainment_vec,'politics':politics_vec,'sport':sport_vec,'tech':tech_vec}

        ###4类
        # category_dict = {'accounts': accounts_c, 'biology': biology_c, 'geography': geography_c,'physics': physics_c}
        # category_all_vec_dict = {'accounts': accounts_vec, 'biology': biology_vec, 'geography': geography_vec, 'physics':physics_vec}

        ###5 abstract
        category_dict = {'Business': Business_c, 'CSAI': CSAI_c, 'Law': Law_c,'Sociology': Sociology_c, 'Trans': Trans_c}
        category_all_vec_dict = {'Business': Business_vec, 'CSAI': CSAI_vec, 'Law': Law_vec, 'Sociology': Sociology_vec, 'Trans': Trans_vec}

        test_dir_list = os.listdir(test_dir)
        # rightNum = 0
        # allNum = 0

        result=''
        # original_num_dict={'business':0,'entertainment':0,'politics':0,'sport':0,'tech':0}###原始正确的标签
        # predict_num_dict={'business':0,'entertainment':0,'politics':0,'sport':0,'tech':0}###预测的标签
        # predict_right_dict={'business':0,'entertainment':0,'politics':0,'sport':0,'tech':0}###预测标签==原始正确的标签
        y_true=[]
        y_pred=[]

        for i in range(len(test_dir_list)):
            # rightNum = 0
            # allNum = 0
            # print(test_dir_list[i])
            # allNum += 1
            right_category = test_dir_list[i]

            second_dir = os.path.join(test_dir, test_dir_list[i])
            second_dir_list = os.listdir(second_dir)
            for j in range(len(second_dir_list)):
                path = os.path.join(second_dir, second_dir_list[j])
                # allNum+=1
                # original_num_dict[right_category] += 1
                y_true.append(right_category)
                # predict_category = classify(path, category_keyword_dict)
                # print('right_category    ', right_category, ';  predict_category    ', predict_category)
                # if right_category == predict_category:
                #     rightNum += 1

                predictFile_keyword_list=readFile(path)
                try:
                    predict_vec=bc.encode(predictFile_keyword_list,is_tokenized=False)
                    predict_c = center(predict_vec)
                    # print(predict_c.shape, '   555')

                    # predict_category =categorize_knn(category_all_vec_dict,predict_vec)
                    predict_category = categorize(category_dict, predict_c, right_category)
                    # predict_num_dict[predict_category] += 1
                    y_pred.append(predict_category)
                except:
                    print('error2333')
                    y_pred.append('politics')

                # predict_c=center(predict_vec)
                # # print(predict_c.shape, '   555')
                #
                # # predict_category =categorize_knn(category_all_vec_dict,predict_vec)
                # predict_category=categorize(category_dict,predict_c,right_category)
                # # predict_num_dict[predict_category] += 1
                # y_pred.append(predict_category)

        print('y_true    ',y_true)
        print('y_pred    ',y_pred)
dir='textrank\\5AbstractsGroup-train'
category_keyword_dict=readCategoryKeyWord(dir)
test_dir='textrank\\5AbstractsGroup-test'
classify(test_dir,category_keyword_dict)
# test_dir_list=os.listdir(test_dir)
# rightNum=0
# allNum=0
# for i in range(len(test_dir_list)):
#     # print(test_dir_list[i])
#     allNum+=1
#     right_category=test_dir_list[i]
#     second_dir=os.path.join(test_dir,test_dir_list[i])
#     second_dir_list=os.listdir(second_dir)
#     for j in range(len(second_dir_list)):
#         path=os.path.join(second_dir,second_dir_list[j])
#         predict_category=classify(path,category_keyword_dict)
#         print('right_category    ',right_category,';  predict_category    ',predict_category)
#         if right_category==predict_category:
#             rightNum+=1