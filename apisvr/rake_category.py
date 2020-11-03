'''extract keywords of documents of one category'''
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
        self.wordFrequency = {}
        self.wordDegree = {}
        self.wordScore = {}
        self.phraseScore = {}
        self.all_cate_wordFrequency = {}
        self.all_cate_wordDegree = {}
        # read documents
        self.docs = []
        for document in codecs.open(inputFilePath, 'r', 'utf-8',errors='ignore'):
            # print('doc111 ',self.docs)
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

    def separateWords(self, text):
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
        # wordFrequency = {}
        # wordDegree = {}
        for phrase in phrases:
            # print('phrase   ',phrase)
            wordList = self.separateWords(phrase)
            print('wordlist   ',wordList)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            print('wordListDegree   ',wordListDegree)
            for word in wordList:
                if word not in self.wordFrequency.keys():
                    self.wordFrequency.setdefault(word, 0)
                    self.wordFrequency[word] += 1
                    self.wordDegree.setdefault(word, 0)
                    self.wordDegree[word] += wordListDegree
                else:
                    self.wordFrequency[word] += 1
                    self.wordDegree[word] += wordListDegree
        for item in self.wordFrequency:
            self.wordDegree[item] = self.wordDegree[item] + self.wordFrequency[item]
            print('wordDegree[item]   ',self.wordDegree[item])

        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        # wordScore = {}
        for item in self.wordFrequency:
            self.wordScore.setdefault(item, 0)

            self.wordScore[item] = self.wordDegree[item] * 1.0 /( self.wordFrequency[item]+self.all_cate_wordFrequency[item])

        for phrase in phrases:
            self.phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += self.wordScore[word]
            self.phraseScore[phrase] = candidateScore
        return self.phraseScore


    def calculateAllFieldPhraseFrequency(self, phrases):

        for phrase in phrases:
            # print('phrase   ',phrase)
            wordList = self.separateWords(phrase)
            print('wordlist   ',wordList)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            print('wordListDegree   ',wordListDegree)
            for word in wordList:
                if word not in self.all_cate_wordFrequency.keys():
                    self.all_cate_wordFrequency.setdefault(word, 0)
                    # self.all_cate_wordFrequency(word, 0)
                    self.all_cate_wordFrequency[word] += 1

                else:
                    self.all_cate_wordFrequency[word] += 1
                    # self.wordDegree[word] += wordListDegree

    def execute(self):

        all_docs = []
        allphrases = []
        file_dir = 'new-experiment-LDA//5AbstractsGroup-train'
        dir_list = os.listdir(file_dir)
        print(dir_list,'   dir222')
        for m in range(len(dir_list)):
            path = os.path.join(file_dir, dir_list[m])
            for document1 in codecs.open(path, 'r', 'utf-8', errors='ignore'):#####document1：一句话
                # print('doc111 ',document1)
                all_docs.append(document1)
                sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
                sentences = sentenceDelimiters.split(document1)
                # print('sentences333   ',sentences)
                phrases = []
                for s in sentences:
                    tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                    phrasesOfSentence = tmp.split("|")
                    for phrase in phrasesOfSentence:
                        phrase = phrase.strip().lower()
                        if phrase != "" and len(phrase) >= self.minPhraseChar and len(
                                phrase.split()) <= self.maxPhraseLength:
                            phrases.append(phrase)
                print('phrases222   ',phrases)
                self.calculateAllFieldPhraseFrequency(phrases)

        allKeyWords={}
        for document in self.docs:

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

            phraseScore = self.calculatePhraseScore(phrases)
            if phraseScore!={}:
                print(phraseScore,'  333')
                keywords = sorted(phraseScore.items(), key=operator.itemgetter(1), reverse=True)
                print(type(keywords))
                allKeyWords.update(keywords)


        print(allKeyWords,'   111')
        print('here      ',sorted(allKeyWords.items(), key=lambda item:item[1], reverse=True))
        sortDict=sorted(allKeyWords.items(), key=lambda item: item[1], reverse=True)

        sortDictCut=dict(sortDict[0:int(len(sortDict) / 3)])
        file = codecs.open(self.outputFilePath, 'w', 'utf-8')
        for i in sortDictCut.keys():
            outline=''
            outline+=i
            outline+=':'
            outline+=str(sortDictCut[i])
            file.write(outline + "\n")
        file.close()


file='new-experiment-LDA//5AbstractsGroup-train'
dir=os.listdir(file)
for i in range(len(dir)):
    path=os.path.join(file,dir[i])
    outPath='new-experiment-LDA/keyword/5AbstractsGroup-train-new/'
    outPath+=dir[i]
    print(outPath)
    rake = Rake(path, 'data/stoplists/SmartStoplist.txt',
                outPath, 1, 2)
    # rake = Rake("5AbstractsGroup_topic_text/Trans.txt",'data/stoplists/SmartStoplist.txt','5AbstractsGroup_topic_keyword/Trans.txt',1,2)
    rake.execute()

