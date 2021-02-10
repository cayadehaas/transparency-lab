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
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
from nltk.chunk import ne_chunk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_model = SentimentIntensityAnalyzer()

nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')


def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=set(),
              verbose=0):
    """
    Run VADER on a sentence from spacy

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -empty set -> all parts of speech are provided
    -non-empty set: only these parts of speech are considered
    :param int verbose: if set to 1, information is printed
    about input and output

    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores

sentiment_analysis = []
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
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            sentiment_analysis.append(run_vader(sentence, lemmatize=True, verbose=1))
            df = pd.read_csv('../TextFiles/reviews.tsv', sep='\t')
            df.head()


df = pd.DataFrame(zip(sentiment_analysis), columns=['sentiment analysis'])
# Create a Pandas Excel writer using XlsxWriter as the engine.

writer = pd.ExcelWriter('sentiment_analysis_whitepapers.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()



