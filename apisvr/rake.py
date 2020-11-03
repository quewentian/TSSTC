'''extract keywords of individual document'''

import re
import operator
import argparse
import codecs
import os

def isNumber(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


class Rake:

    def __init__(self, inputFilePath, stopwordsFilePath, outputFilePath, minPhraseChar, maxPhraseLength):
        self.outputFilePath = outputFilePath
        self.minPhraseChar = minPhraseChar
        self.maxPhraseLength = maxPhraseLength
        # read documents
        self.docs = []      ###self.docs: content of one document
        for document in codecs.open(inputFilePath, 'r', 'utf-8',errors='ignore'):
            print('doc111 ',document)
            self.docs.append(document)
        # read stopwords
        stopwords = []
        for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
            stopwords.append(word.strip())
        stopwordsRegex = []
        for word in stopwords:
            regex = r'\b' + word + r'(?![\w-])'
            stopwordsRegex.append(regex)
        self.stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)

    def separateWords(self, text):###split sentences into phrases
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(word) > 0 and word != '' and not isNumber(word):
                words.append(word)
        return words

    def calculatePhraseScore(self, phrases):
        # calculate wordFrequency and wordDegree
        wordFrequency = {}
        wordDegree = {}
        for phrase in phrases:
            # print('phrase   ',phrase)###phrase:cutted phrases
            wordList = self.separateWords(phrase)
            print('wordlist   ',wordList)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            print('wordListDegree   ',wordListDegree)
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]
            print('wordDegree[item]   ',wordDegree[item])

        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]

        # calculate phraseScore
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += wordScore[word]
            phraseScore[phrase] = candidateScore
        return phraseScore

    def execute(self):

        allKeyWords={}
        for document in self.docs:
            print('docu333   ',document)
            # split a document into sentences
            sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
            sentences = sentenceDelimiters.split(document)
            # generate all valid phrases
            phrases = []
            for s in sentences:
                tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                phrasesOfSentence = tmp.split("|")
                for phrase in phrasesOfSentence:
                    phrase = phrase.strip().lower()
                    if phrase != "" and len(phrase) >= self.minPhraseChar and len(
                            phrase.split()) <= self.maxPhraseLength:
                        phrases.append(phrase)

            # calculate phrase score
            phraseScore = self.calculatePhraseScore(phrases)
            if phraseScore!={}:
                print(phraseScore,'  333')
                keywords = sorted(phraseScore.items(), key=operator.itemgetter(1), reverse=True)
                print(type(keywords))
                allKeyWords.update(keywords)


        print(allKeyWords,'   111')
        print('here      ',sorted(allKeyWords.items(), key=lambda item:item[1], reverse=True))
        sortDict=sorted(allKeyWords.items(), key=lambda item: item[1], reverse=True)
        if len(sortDict)>=50:
            sortDictCut = dict(sortDict[0:49])
        else:
            sortDictCut = dict(sortDict)

        file = codecs.open(self.outputFilePath, 'w', 'utf-8')
        for i in sortDictCut.keys():
            outline=''
            outline+=i
            outline+=':'
            outline+=str(sortDictCut[i])
            file.write(outline + "\n")
        file.close()

file_dir='new-experiment-LDA/5AbstractsGroup-test'
dir_list=os.listdir(file_dir)
for i in range(len(dir_list)):
    second_dir=os.path.join(file_dir,dir_list[i])
    dir_list2=os.listdir(second_dir)
    for j in range(len(dir_list2)):
        path=os.path.join(second_dir,dir_list2[j])
        outpath=path.replace('5AbstractsGroup-test','keyword/5AbstractsGroup-test-new')
        print(outpath,'   out222')

        rake = Rake(path,'data/stoplists/SmartStoplist.txt',outpath,1,2)
        rake.execute()