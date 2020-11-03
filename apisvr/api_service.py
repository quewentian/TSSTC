# #!/usr/bin/env python3
# # coding:utf-8
#
# """
#  @Time    : 2019/2/28 15:31
#  @Author  : xmxoxo (xmhexi@163.com)
#  @File    : tran_service.py
# """
import argparse
import flask
import logging
import json
import os
import re
import sys
import string
import time
import numpy as np
import math
import pandas as pd
# from bert_base.client import BertClient
from client.bert_serving.client import BertClient

def dotProduct(v1, v2):###点乘
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    z = v1 * v2.T
    return z

def center(vec):###球心
    c = np.zeros((1, 768))
    for i in range(len(vec)):
        c += vec[i]
    result=c / len(vec)
    print(result.shape)
    return result

def categorize(category,vec):###分类
    max_cos=0   #最大的相似度
    max_cate='' #最大相似度的类
    for key in category:
        # print(category[key])
        # print(category[key].shape)
        # print(vec.shape)
        cosine_sim = dotProduct(vec, category[key]) / math.sqrt(dotProduct(vec, vec) * dotProduct(category[key], category[key]))
        print('每类得分   ',key,'：   ',cosine_sim[0][0])
        if cosine_sim[0][0]>max_cos:
            max_cos=cosine_sim[0][0]
            max_cate=key
    return max_cate



with BertClient(ip='10.0.11.212', port=5555, port_out=5556, show_server_config=True, timeout=1000) as bc:
#         start_t =
#         print('list_text:  ',list_text)time.perf_counter()
#         result = bc.encode(list_text, is_tokenized=False)
#         rst = bc.encode(['Text classification aims at mapping documentsinto  a  set  of  predefined  categories. Supervised  machine  learning  models  have  showngreat  success  in  this  area  but  they  require  alarge  number  of  labeled  documents  to  reachadequate  accuracy',
#                          'Computer vision','Speech Recognition','big data','machine learning','robot'],is_tokenized=False)
# #         print(rst)
#         print(rst.shape)#<class 'numpy.ndarray'>[3,768]，一个句子768维
        # print(rst[0])
#         # print(rst[1])
#         # print(rst[2])
#         nlp=['NLP','entity extraction','relation extraction','ontology','domain knowledge','Machine translation,Discourse', 'dialogue and pragmatics',
#              'Natural language generation','Speech recognition','Lexical semantics','Phonology',
#              'morphology','Natural language processing','Computational Linguistics','Text classification ',
#              'Semantic representation','Word vector','Word embedding','Pre-training',
#              'Natural language understanding','Natural language generation','machine translation',
#              'Question Answering','Information retrieval','Text mining','Machine reading comprehension' ]
        nlp = ['NLP', 'entity extraction', 'relation extraction', 'ontology', 'domain knowledge',
       'Machine translation,Discourse', 'dialogue and pragmatics',
       'Natural language generation', 'Speech recognition', 'Lexical semantics', 'Phonology',
       'morphology', 'Natural language processing', 'Computational Linguistics', 'Text classification ',
       'Word vector', 'Word embedding', 'Pre-training',
       'Natural language understanding', 'Natural language generation', 'machine translation',
       'Question Answering', 'Information retrieval', 'Text mining', 'Machine reading comprehension']
        vec=bc.encode(nlp,is_tokenized=False)
        print(vec.shape)
        c_nlp=center(vec)
        # cv=['Computer vision','Image Identification','Image generation','Target Detection',
        #     'Face recognition','Graphic understanding','image segment','3D','Action and behavior recognition'
        #     ,'Adversarial learning', 'adversarial attack and defense methods,Biometrics',
        #     'face, gesture, body pose,Computational photography', 'image synthesis','Image retrieval',
        #     'Image retrieval','GAN', 'Motion and Tracking,Segmentation and Grouping', 'Object Recognition',
        #     'Object Detection and Categorization','Gesture Analysis','Medical Image Analysis',
        #     'Vision for Graphics']
        cv=['Computer vision','Image Identification','Image generation','Target Detection',
        'Face recognition','Graphic understanding','image segment','3D','2D','Action and behavior recognition'
        ,'Adversarial learning', 'adversarial attack and defense methods',
        'face, gesture, body pose,Computational photography', 'image synthesis','Image retrieval',
        'Image retrieval','GAN', 'Motion and Tracking,Segmentation and Grouping', 'Object Recognition',
        'Object Detection','Gesture Analysis','Image Analysis','Vision for Graphics']
        vec_cv=bc.encode(cv,is_tokenized=False)
        c_cv=center(vec_cv)
        # sr=['Speech synthesis','Acoustic model','voice signal','Voiceprint recognition','crying detection',
        #     'Audio event detection','Oral comprehension','Audio interactive','voice wakeup',
        #     'Intelligent voice,voice assistant','tone','Audio','resonance','sound','ultrasound']
        sr=['Speech synthesis','Acoustic model','voice signal','Audio event detection',
            'Oral comprehension','Audio interactive','Audio and Acoustic Signal Processing',
            'Remote Sensing and Signal Processing','Speech Processing',
            ]
      #   sr = ['Speech synthesis', 'Acoustic model',
      #   'Oral comprehension', 'Audio interactive', 'Audio and Acoustic Signal Processing',
      # 'Remote Sensing and Signal Processing', 'Speech Processing'
      # ]
        vec_sr=bc.encode(sr,is_tokenized=False)
        c_sr=center(vec_sr)
        vp = ['Video processing','Video processing','Visual tracking','video tagging','video generation',
              'Intelligent video analysis','target detection','Pedestrian tracking','human pose estimation']
        vec_vp = bc.encode(vp, is_tokenized=False)
        c_vp = center(vec_vp)
        category_dict={'NLP':c_nlp,'CV':c_cv,'SR':c_sr,'VP':c_vp}

        path='test.xls'
        title = pd.read_excel(path, sheet_name = 0,usecols=[1])
        category=pd.read_excel(path, sheet_name = 0,usecols=[0])
        # print(df)
        # print(type(df))
        np_title=np.array(title)###dataframe格式转为np array格式
        np_category=np.array(category)###dataframe格式转为np array格式
        nlp_real=0
        nlp_pred=0
        cv_real=0
        cv_pred=0
        sr_real=0
        sr_pred=0
        for i in range(len(np_title)):
            text=np_title[i].tolist()
            rst = bc.encode(text,is_tokenized=False)
#         print(rst)

            max_cate=categorize(category_dict,rst)
            print('实际结果：   ',np_category[i][0],'；   预测结果：   ',max_cate)
            if np_category[i][0]=='NLP':
                nlp_real+=1
            if np_category[i][0]==max_cate and np_category[i][0]=='NLP':
                nlp_pred+=1
            if np_category[i][0] == 'CV':
                cv_real += 1
            if np_category[i][0] == max_cate and np_category[i][0] == 'CV':
                cv_pred += 1
            if np_category[i][0] == 'SR':
                sr_real += 1
            if np_category[i][0] == max_cate and np_category[i][0] == 'SR':
                sr_pred += 1
        print(nlp_pred)
        print(cv_pred)
        print(sr_pred)

        print("自然语言处理： ",nlp_pred/nlp_real)
        print("计算机视觉： ",cv_pred / cv_real)
        print("语音识别： ",sr_pred / sr_real)

        # cosine_sim = dotProduct(rst[0], rst[1])/math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(rst[1], rst[1]))
        # print('nlp/cv   ',cosine_sim)
        # cosine_sim2 = dotProduct(rst[0], rst[2])/math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(rst[2], rst[2]))
        # print('nlp/sr   ',cosine_sim2)
        # cosine_sim3 = dotProduct(rst[0], rst[3]) / math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(rst[3], rst[3]))
        # print('nlp/big data   ', cosine_sim3)
        # cosine_sim4 = dotProduct(rst[0], rst[4]) / math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(rst[4], rst[4]))
        # print('nlp/ml   ', cosine_sim4)
        # cosine_sim5 = dotProduct(rst[0], rst[5]) / math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(rst[5], rst[5]))
        # print('nlp/robot   ', cosine_sim5)
        # cosine_sim6 = dotProduct(rst[0], c_nlp) / math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(c_nlp, c_nlp))
        # print('nlp/c_nlp   ', cosine_sim6)
        # cosine_sim7 = dotProduct(rst[0], c_cv) / math.sqrt(dotProduct(rst[0], rst[0]) * dotProduct(c_cv, c_cv))
        # print('nlp/c_cv   ', cosine_sim7)


#         # rst = bc.encode([['second', 'dissociation', 'constant','have','been','measured','spectrophotometrically',
#         #                   'of', 'phenyl','benoates'],
#         #                    ['The', 'PK', 'of', 'amino','acid','is','4.2'],
#         #                    ['then', 'do', 'it', 'better']], is_tokenized=True)

#
#
# # 切分句子
# def cut_sent(txt):
#     # 先预处理去空格等
#     txt = re.sub('([　 \t]+)', r" ", txt)  # blank word
#     txt = txt.rstrip()  # 段尾如果有多余的\n就去掉它
#     nlist = txt.split("\n")
#     nnlist = [x for x in nlist if x.strip() != '']  # 过滤掉空行
#     return nnlist
#
# ##将输入的文本的标点符号给取出来
# def add_labels(list_text):
#     sentence_labels_list = []###所有句子的带有标点的嵌套list  [[句子1.。。],[句子2.。。],...]
#     labels = ['.', ',', '!', '?', '\'']
#     for i in range(len(list_text)):
#         # print(list[i].split())
#         # print(list[i][len(list[i])-1])
#         # print('i=   ',i)
#         # print('list_text[i][len(list_text[i]) - 1]     ',list_text[i][len(list_text[i]) - 1])
#         strip_sentence=list_text[i].strip()##去掉句子里的空白的字符
#         # print(strip_sentence)
#         strip_sentence_list=strip_sentence.split()
#         one_sentence_list=[]  ##一句话的list
#         for j in range(len(strip_sentence_list)):
#             if strip_sentence_list[j][len(strip_sentence_list[j])-1] in labels:###单词是末尾
#                 # print(strip_sentence_list[j])
#                 # print(strip_sentence_list[j][0:len(strip_sentence_list[j])-1])##单词
#                 # print(strip_sentence_list[j][len(strip_sentence_list[j])-1])##标点
#                 one_sentence_list.append(strip_sentence_list[j][0:len(strip_sentence_list[j])-1])##最后的单词
#                 one_sentence_list.append(strip_sentence_list[j][len(strip_sentence_list[j])-1])##句末标点
#             else:##非句子末尾
#                 one_sentence_list.append(strip_sentence_list[j])
#         sentence_labels_list.append(one_sentence_list)
#     return sentence_labels_list
#
# # 对句子进行预测识别
# def ner_pred(list_text):
#     # 文本拆分成句子
#     # list_text = cut_sent(text)
#     print("total setance: %d" % (len(list_text)))
#     print(list_text,'   list_text')
#     with BertClient(ip='10.0.11.212', port=5575, port_out=5576, show_server_config=False, check_version=False,
#                     check_length=False, timeout=1000, mode='NER') as bc:
#         start_t =
#         print('list_text:  ',list_text)time.perf_counter()
#         result = bc.encode(list_text, is_tokenized=False)
#         # rst = bc.encode(['second dissociation constant have been measured spectrophotometrically',
#         #                  'the PK of amino acid is 4.2','hello world sir'],is_tokenized=False)
#         # rst = bc.encode([['second', 'dissociation', 'constant','have','been','measured','spectrophotometrically',
#         #                   'of', 'phenyl','benoates'],
#         #                    ['The', 'PK', 'of', 'amino','acid','is','4.2'],
#         #                    ['then', 'do', 'it', 'better']], is_tokenized=True)
#         # rst = bc.encode(['李克强访问国务院','你一定可以考公务员进入公安厅','我们家住在钓鱼台'])
#         print('result:', result)##result: [list(['<O>', '<O>', '<O>']) list(['<O>', '<O>', '<O>', '<O>', '<O>'])
#         print('time used:{}'.format(time.perf_counter() - start_t))
#     ######返回结构为：
#     # rst: [{'pred_label': ['0', '1', '0'], 'score': [0.9983683228492737, 0.9988993406295776, 0.9997349381446838]}]
#     ######抽取出标注结果
#     # pred_label = rst[0]["pred_label"]
#     # pred_label=rst
#     # result_txt = [[pred_label[i], list_text[i]] for i in range(len(pred_label))]
#     result_txt = [result[i] for i in range(len(result))]
#     # add_labels_txt = add_labels(list_text)
#     result_dict = {}
#     for i in range(len(result)):
#         # print(result[i])   ###将[{}]中，被[]括起来的{}拿出来
#         result_dict = result[i]
#     # print(result_txt,'   result_txt')
#     # res={}
#     # res['result'] = result_txt
#     # res['txt'] = add_labels_txt
#     # print('res:    ',res)
#     return result_dict
#     ##################################测试代码################################
#     # rst=[['<O>', '<O>', '<O>'],['<O>', '<O>', '<O>', '<O>'],['<O>', '<O>', '<O>', '<O>']]
#     # # list_text=['Hello world!', 'I love learn chemistry!', 'Chemical bond exergy is essential.']
#     # print(rst)
#     # pred_label = rst
#     # result_txt = [[pred_label[i], list_text[i]] for i in range(len(pred_label))]
#     # print(result_txt,'   result_txt')
#     # return result_txt
#
#
# def flask_server(args):
#     pass
#     from flask import Flask, request, render_template, jsonify
#
#     app = Flask(__name__)
#
#     # from app import routes
#
#     @app.route('/')
#     def index():
#         return render_template('index.html', version='V 0.1.2')
#
#     @app.route('/api/v0.1/query', methods=['POST'])
#     def query():
#         res = {}
#         txt = request.values['text']
#         if not txt:
#             res["result"] = "error"
#             return jsonify(res)
#         lstseg = cut_sent(txt)
#         print('-' * 30)
#         print('结果,共%d个句子:' % (len(lstseg)))
#         for x in lstseg:
#             print("第%d句：【 %s】" % (lstseg.index(x), x))
#         print('-' * 30)
#         if request.method == 'POST' or 1:
#             result_dict = ner_pred(lstseg)
#         print('result:    ',result_dict)
#         final=addTokenColors(result_dict['tokens'],result_dict['pred_label'])
#         # final=addColor(result)
#         # print('result:%s' % str(result))
#         # res='O O O O O O O O'
#         # print(res)
#         # return jsonify(res)
#         return final
#
#     app.run(
#         host=args.ip,  # '0.0.0.0',
#         port=args.port,  # 8910,
#         debug=True
#     )
#
#
# def main_cli():
#     pass
#     parser = argparse.ArgumentParser(description='API demo server')
#     parser.add_argument('-ip', type=str, default="10.0.11.38",
#                         help='chinese google bert model serving')
#     parser.add_argument('-port', type=int, default=8910,
#                         help='listen port,default:8910')
#
#     args = parser.parse_args()
#
#     flask_server(args)
#
#
# if __name__ == '__main__':
#     main_cli()