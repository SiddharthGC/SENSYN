#Sentence feature extractor
#Author: Siddharth Chandrasekar, Last Modified: 11/30/19 15:09 CST

import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
nlp = spacy.load('en_core_web_sm')

#Synset data structure
class SynsetStruct():
    def __init__(self, words):
        self.words = words
        self.synsets = []
        for iword in self.words:
            self.synsets.append(wn.synsets(iword))

#Feature extraction class
class FeatureExtraction:
    def __init__(self, sentence):
        self.sentence = sentence
        self.wordlist = word_tokenize(self.sentence)

    def removeStopWords(self):
        self.wordlist = [word for word in self.wordlist if word not in stopwords.words('english')]

    def getWordList(self):
        return self.wordlist

    def getPosTags(self):
        return nltk.pos_tag(self.wordlist)

    def getDependencyParseTree(self):
        return nlp(self.sentence)

    def getLemmas(self):
        doc = nlp(self.sentence)
        LemmaList = []
        for wordx in doc:
            LemmaList.append(wordx.lemma_)
        return LemmaList

    def getSynsets(self):
        syn = SynsetStruct(self.wordlist)
        return syn

    def getActiveOrPassive(self):
        beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']
        aux = ['do', 'did', 'does', 'have', 'has', 'had']
        tokens = nltk.pos_tag(self.wordlist)
        tags = [i[1] for i in tokens]
        if tags.count('VBN') == 0:
            return "Active"
        elif tags.count('VBN') == 1 and 'been' in self.wordlist:
            return "Active"
        else:
            pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and self.wordlist[i] != 'been']
            for end in pos:
                chunk = tags[:end]
                start = 0
                for i in range(len(chunk), 0, -1):
                    last = chunk.pop()
                    if last == 'NN' or last == 'PRP':
                        start = i
                        break
                sentchunk = self.wordlist[start:end]
                tagschunk = tags[start:end]
                verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')]
                if verbspos != []:
                    for i in verbspos:
                        if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:
                            break
                    else:
                        return "Passive"
        return "Active"

#Console application to show as a demo

print("Enter the input sentences: ")
x = input()
tokenized = sent_tokenize(x)
for i in tokenized:
    print('Sentence "' + i + '":')
    f1 = FeatureExtraction(i)
    print('Words : ')
    print(f1.getWordList())
    print('\n')
    print('POS Tags : ')
    print(f1.getPosTags())
    print('\n')
    print('Dependency parse tree: ')
    doc = f1.getDependencyParseTree()
    for token in doc:
        print("{2}({3}-{6}, {0}-{5})".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1))
    print('\n')
    print('Lemmas : ')
    print(f1.getLemmas())
    print('\n')
    print('Synsets : ')
    synObj = f1.getSynsets()
    for word in synObj.words:
        for synsets in synObj.synsets:
            for synset in synsets:
                print("Lemmas: ", synset.lemma_names())
                print("meronyms: ", synset.part_meronyms())
                print("holonyms: ", synset.part_holonyms())
                print("hypernyms:", synset.hypernyms())
                print("hyponyms:", synset.hyponyms())
    print('\n')
    print('Voice : '+ f1.getActiveOrPassive())
    print('\n')
    f1.removeStopWords()
    print('Without stop words :', f1.wordlist)
    print('\n')