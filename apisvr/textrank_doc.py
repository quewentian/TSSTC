"""extract keywords for documents with textrank"""

import TextRank
import codecs
import os

def textRank(inpath,outpath):
    string = codecs.open(inpath, 'r', 'utf-8',errors='ignore').read()
    textrank_results = TextRank.extractKeyphrases(string)
    sorted_keywords = sorted(textrank_results.items(), key=lambda x: x[1], reverse=True)
    print(sorted_keywords)
    outString=''
    for i in range(len(sorted_keywords)):
        print(sorted_keywords[i])
        print(sorted_keywords[i][0])
        print(sorted_keywords[i][1])
        outString+=sorted_keywords[i][0]
        outString+=':'
        outString+=str(sorted_keywords[i][1])
        outString+='\n'

    # out_path='output/5AbstractsGroup-test1/Business/0401.txt'
    with open(outpath,'w',encoding='utf-8') as f:
        f.write(outString)

data_path='myData//5AbstractsGroup-test'
data_dir=os.listdir(data_path)
# second_path=os.path.join(data_path,data_dir[i])
for i in range(len(data_dir)):
    second_path=os.path.join(data_path,data_dir[i])
    print(second_path)
    second_dir=os.listdir(second_path)
    for j in range(len(second_dir)):
        third_path=os.path.join(second_path,second_dir[j])
        # print(third_path)
        outpath=third_path.replace('myData','textrank_output')
        print('input   ',third_path,"    output   ",outpath)
        textRank(third_path,outpath)