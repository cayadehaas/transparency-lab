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
import csv
from tqdm import tqdm
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
        pass
        # print()
        # print('INPUT SENTENCE', sent)
        # print('INPUT TO VADER', input_to_vader)
        # print('VADER OUTPUT', scores)

    return scores

compound_sentences = []
verhaallijn = []

n1 = '[0-9]+ years'              #50 years
n2 = ' [1-2][0-9][0-9][0-9] '     #2020
n7 = '[0-9]+\.[0-9]+ +[a-z]+'  # 1.1 billion
n8 = '[0-9]+\,[0-9]+ +[a-z]+'  # 5,000 people
n9 = '[0-9]+\.[0-9]+ +[a-z]+'  # 5.000 people
n10 = '[0-9]+ +[a-z]+'  # 15 respondents
n11 = '\+1 [0-9]+ +[0-9]+ +[0-9]+ +[a-z]+' #+1 630 258 2402 w+
n12 = '\+[0-9]+ [0-9]+-[0-9]+-[0-9] +[a-z]+'#+49 69-273992-0 w+
n13 = '\+[0-9]+ [0-9]+ [0-9]+ [0-9]+ +[a-z]+' #+44 370 904 3601 w+
n14 = '\+[0-9]+ +[0-9]+ [a-z]+' #+49 692739920 w+
n15 = '\+ [0-9]+ +[0-9]+ +[0-9]+ +[a-z]+' #+ 49 69 2739920 w+

with open('verhaallijn_sentiment_analysis.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["FILENAME", 'VERHAALLIJN', 'NUMBERS', 'VRAAGTEKENS', '# VRAAGTEKENS'])
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/whitepapers_test', topdown=False):
        directory = root.split('/')
        english_stopwords = stopwords.words('english')
        for entry in tqdm(files):
            pdf = open('whitepapers_test/' + entry, 'rb')
            rawText = parser.from_file('whitepapers_test/' + entry)
            rawList = rawText['content'].splitlines()
            while '' in rawList: rawList.remove('')
            while ' ' in rawList: rawList.remove(' ')
            while '\t' in rawList: rawList.remove('\t')
            text = ''.join(rawList)
            vraagtekens = re.compile('[?]').findall(text)
            VRAAGTEKENS = len(vraagtekens)
            new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # adds whitespace after between small and captial letter
            good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', new_sentences) #adds whitespace after . and ,
            normal_sentences = re.sub(r'\\xa0', r' ', good_sentences) #changes \xa0 to whitespace
            great_sentences = re.sub(r'[-]', r'', normal_sentences)
            free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2', great_sentences) #adds whitespace between number and letters
            links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", free_sentences) #remove space between link
            links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link

            sentences = nltk.sent_tokenize(links2)
            percentage_list = []
            vraagteken_list = []
            for sentence in sentences:
                numbers = ['percent ', '%']
                compound_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['compound'])
                #check for numbers, percent or % voorkomt in sentence, add to list
                number_list = re.compile("(%s|%s|%s|%s)" % (n7, n8, n9, n10)).findall(str(sentence))
                years = re.compile("(%s|%s)" % (n1, n2)).findall(str(sentences))
                phone_numbers = re.compile("(%s|%s|%s|%s|%s)" % (n11, n12, n13, n14, n15)).findall(str(sentence))


                for year in years[:]: #remove whitespace
                    new_year = year.strip()
                    years.remove(year)
                    years.append(new_year)

                for number in phone_numbers[:]:
                    new_number = number.strip('-')
                    # phone_numbers.remove(number)
                    phone_numbers.append(new_number)

                vraagtekens = re.compile('[?]').findall(sentence)
                if len(vraagtekens) != 0:
                    vraagteken_list.append('2') #wel vraagteken in zin
                else:
                    vraagteken_list.append('1') #geen vraagteken in zin

            for number in number_list[:]: #remove years from number list
                for element in years:
                    if element in number:
                        number_list.remove(number)

            for number in phone_numbers:  # remove phone numbers from number list
                for element in number_list[:]:
                    if element in number:
                        number_list.remove(element)

            for sentence in sentences:
                if len(number_list) != 0:
                    percentage_list.append('2')
                elif any(x in sentence for x in numbers):
                    percentage_list.append('2')
                else:
                    percentage_list.append('1')

            for score in compound_sentences:
                if score >= 0.05:       #positive
                    verhaallijn.append(3)
                elif score <= - 0.05:   #negative
                    verhaallijn.append(1)
                else:                   #neutral
                    verhaallijn.append(2)

            percentage_list = (",".join(percentage_list))  # remove brackets and whitespace from list
            vraagteken_list = (",".join(vraagteken_list))
            verhaallijn = (",".join(repr(e) for e in verhaallijn))

            breakpoint()
            writer.writerow([entry, verhaallijn, percentage_list, vraagteken_list, VRAAGTEKENS])
            compound_sentences = []  # make list empty for next file
            verhaallijn = []  # make list empty for next file
            percentage_list = [] #make list empty for next file


read_file = pd.read_csv(r'verhaallijn_sentiment_analysis.csv')
read_file.to_excel(r'verhaallijn_sentiment_whitepapers.xlsx', index=None, header=True)



