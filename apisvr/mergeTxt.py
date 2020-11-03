import codecs
import os

file='newExperiment/5AbstractsGroup-test'
file_dir=os.listdir(file)
# print(dir)
for k in range(len(file_dir)):
    print(file_dir[k],'   2级目录')
    file_2 = os.path.join(file,file_dir[k])
    dir=os.listdir(file_2)
    content=''
    for i in range(len(dir)):
        path=os.path.join(file_2,dir[i])
        print(path,'  path')  ##newExperiment/accounts

        with codecs.open(path,encoding='utf-8',errors='ignore') as f:
            content+=f.read()
    #
    outPath='new-experiment-LDA/5AbstractsGroup-test/'
    outPath+=file_dir[k]
    outPath+='.txt'
    with open(outPath,'w',encoding='utf-8') as f1:
        f1.write(content)
