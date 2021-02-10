import os
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import nltk
from nltk.corpus import stopwords
import re
from tika import parser
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')

for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    directory = root.split('/')
    english_stopwords = stopwords.words('english')
    for entry in files:
        pdf = open('more_whitepapers/' + entry, 'rb')
        rawText = parser.from_file('more_whitepapers/' + entry)
        try:
            rawList = rawText['content'].splitlines()
            while '' in rawList: rawList.remove('')
            while ' ' in rawList: rawList.remove(' ')
            while '\t' in rawList: rawList.remove('\t')
            text = ''.join(rawList)
            breakpoint()
            text = text.lower()
            tokenized_text = nltk.word_tokenize(text)
            doc = nlp(text)
            pprint([(X.text, X.label_) for X in doc.ents])

            # part_of_speech = nltk.pos_tag(tokenized_text)
            # pattern = 'NP: {<DT>?<JJ>*<NN>}'
            # cp = nltk.RegexpParser(pattern)
            # cs = cp.parse(part_of_speech)
            # print(cs)
            # iob_tagged = tree2conlltags(cs)
            # pprint(iob_tagged)

            # ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
            # print(ne_tree)
            breakpoint()
        except:
            continue
