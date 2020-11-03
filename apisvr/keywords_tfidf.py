'''keywords extracted by TF-IDF'''
import sys,codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l

def get_wordnet_pos(tag):
    """
    获取单词的词性
    :param tag: 词性
    :return: 词类型
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def stemming(content):
    """
    词干提取stemming
    :param content: 文件内容
    :return: 文件内容
    """
    stemmer = SnowballStemmer("english")  # 选择目标语言为英语
    all_words = content.split(' ')
    new_content = []
    for word in all_words:
        new_word = stemmer.stem(word.lower())  # Stem a word 并且转化为小写
        if new_word != ' ':
            new_content.append(new_word)
    return " ".join(new_content)

def lemmatization(content):
    """
    词形还原 lemmatization
    :param content: 文件内容
    :return: 文件内容
    """
    all_words = word_tokenize(content)  # 分词
    tagged_sent = pos_tag(all_words)  # 获取单词词性

    wnl = WordNetLemmatizer()
    new_content = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        new_content.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    return " ".join(new_content)

#根据窗口，构建每个节点的相邻节点,返回边的集合
def createNodes(word_list):
    edge_dict = {}  ####记录节点的边连接字典
    # word_list = []
    window=3
    tmp_list = []
    word_list_len = len(word_list)
    for index, word in enumerate(word_list):
        if word not in edge_dict.keys():
            tmp_list.append(word)
            tmp_set = set()
            left = index - window + 1#窗口左边界  这里是index
            right = index + window#窗口右边界
            if left < 0: left = 0
            if right >= word_list_len: right = word_list_len
            for i in range(left, right):
                if i == index:
                    continue
                tmp_set.add(word_list[i])
            edge_dict[word] = tmp_set
    print('edge_dict    ',edge_dict)
    return edge_dict

    # 根据边的相连关系，构建矩阵
def createMatrix(word_list,edge_dict):
    print(len(set(word_list)),'   word_list大小')
    matrix = np.zeros([len(set(word_list)), len(set(word_list))])
    print('matrix 大小   ',matrix.shape)
    word_index = {}  # 记录词的index
    index_dict = {}  # 记录节点index对应的词

    for i, v in enumerate(set(word_list)):
        word_index[v] = i
        index_dict[i] = v
    for key in edge_dict.keys():
        for w in edge_dict[key]:
            matrix[word_index[key]][word_index[w]] = 1
            matrix[word_index[w]][word_index[key]] = 1
    # 归一化
    for j in range(matrix.shape[1]):
        sum = 0
        for i in range(matrix.shape[0]):
            sum += matrix[i][j]
        for i in range(matrix.shape[0]):
            matrix[i][j] /= sum
    print(matrix)
    return matrix,index_dict

#根据textrank公式计算权重
def calPR(word_list,matrix):
    alpha=0.85
    iternum=500
    PR = np.ones([len(set(word_list)), 1])
    for i in range(iternum):
        PR = (1 - alpha) + alpha * np.dot(matrix, PR)##PR=score
    print('PR     ',PR)
    return PR

#输出词和相应的权重
def sortWeight(index_dict,PR):
    word_pr = {}
    for i in range(len(PR)):
        word_pr[index_dict[i]] = PR[i][0]
    res = sorted(word_pr.items(), key = lambda x : x[1], reverse=True)
    # print("phrase 权重    ",res)
    return res

def getPhrases(lemmatized_text,stopwords_plus):
    phrases = []
    phrase = " "
    for word in lemmatized_text:
        if word in stopwords_plus:
            if phrase != " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase += str(word)
            phrase += " "
    print("Partitioned Phrases (Candidate Keyphrases): \n")
    print(phrases)

# tf-idf获取文本top10关键词
def getKeywords_textrank(data_path,stopwordsFilePath,topK,output_dir):
    corpus0 = []
    stopwords=[]
    for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
        stopwords.append(word.strip())
    # print(stopwords)
    stopwordsRegex = []
    for word in stopwords:
        try:
            regex = r'\b' + word + r'(?![\w-])'
        except:
            print(word,'   error')
        stopwordsRegex.append(regex)
        stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)

    data_dir = os.listdir(data_path)
    doc_name_list=[]
    for i in range(len(data_dir)):
        doc_name_list.append(data_dir[i])
        txt_path=os.path.join(data_path,data_dir[i])
        print(txt_path)
        with codecs.open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            doc=f.read()
            tmp = re.sub(stopwordsPattern, ' ', doc.strip())
            tmp=re.sub("[\\\'\t\n\r\.\!\/_,$%^*()\[\]+\"<>\-:]+|[|+——！，。？?、~@#￥%……&*（）]+", " ", tmp)
            corpus0.append(tmp)

    corpus=[]
    for j in range(len(corpus0)):
        aftlemmatization=lemmatization(corpus0[j])
        # print('aft222   ',aftlemmatization)

        aftlemmatization_list=aftlemmatization.split()###word_list
        # print('aftlemmatization_list   ',aftlemmatization_list)
        edge_dict=createNodes(aftlemmatization_list)
        matrix,index_dict=createMatrix(aftlemmatization_list,edge_dict)
        PR=calPR(aftlemmatization_list,matrix)
        sortWordList=sortWeight(index_dict,PR)###已经找出来权值最大的词，再去找对应的词组

        getPhrases(aftlemmatization_list,stopwords)



def main():
    # 读取数据集

    data_path='myData/5AbstractsGroup-test1/Business'
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('data/stop_words.txt', 'r',encoding='utf-8',errors='ignore').readlines()]
    output_dir = 'output/5AbstractsGroup-test1/Business'
    # tf-idf关键词抽取
    result = getKeywords_textrank(data_path,'data\SmartStoplist.txt',500,output_dir)


if __name__ == '__main__':
    main()
