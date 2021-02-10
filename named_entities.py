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
from nltk.chunk import ne_chunk

nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')
result = []
counter = []
for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    directory = root.split('/')
    english_stopwords = stopwords.words('english')
    for entry in files:
        pdf = open('more_whitepapers/' + entry, 'rb')
        rawText = parser.from_file('more_whitepapers/' + entry)
        rawList = rawText['content'].splitlines()
        while '' in rawList: rawList.remove('')
        while ' ' in rawList: rawList.remove(' ')
        while '\t' in rawList: rawList.remove('\t')
        text = ''.join(rawList)
        tokenized_text = nltk.word_tokenize(text)
        doc = nlp(text)
        result.append([(X.text, X.label_) for X in doc.ents])
        labels = [x.label_ for x in doc.ents]
        counter.append(Counter(labels))

df = pd.DataFrame(zip(result, counter), columns=['named entity', 'counter'])
# Create a Pandas Excel writer using XlsxWriter as the engine.

writer = pd.ExcelWriter('named_entities_whitepapers.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()




