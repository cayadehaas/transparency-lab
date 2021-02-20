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
neutral_sentences = []
negative_sentences = []
positive_sentences = []

with open('sentiment_analysis.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["FILENAME", "NBR. OF POSITIVE SENTENCES", "NBR. OF NEGATIVE SENTENCES", "NBR. OF NEUTRAL SENTENCES"])
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
        directory = root.split('/')
        english_stopwords = stopwords.words('english')
        for entry in tqdm(files):
            pdf = open('more_whitepapers/' + entry, 'rb')
            rawText = parser.from_file('more_whitepapers/' + entry)
            rawList = rawText['content'].splitlines()
            while '' in rawList: rawList.remove('')
            while ' ' in rawList: rawList.remove(' ')
            while '\t' in rawList: rawList.remove('\t')
            text = ''.join(rawList)

            good_sentences = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text) #adds whitespace after . and ,
            normal_sentences = re.sub(r'\\xa0', r' ', good_sentences) #changes \xa0 to whitespace
            great_sentences = re.sub(r'[-]', r'', normal_sentences)
            new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', great_sentences) #adds whitespace after between small and captial letter
            free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2', new_sentences)
            sentences = nltk.sent_tokenize(free_sentences)

            table = {ord(char): '' for char in string.punctuation}
            cleaned_messy_sentence = []
            for messy_word in sentences:
                cleaned_word = messy_word.translate(table)
                cleaned_messy_sentence.append(cleaned_word)

            without_stopwords = []
            for token in cleaned_messy_sentence:
                if token not in english_stopwords:
                    without_stopwords.append(token)
            while '' in without_stopwords: without_stopwords.remove('')

            for sentence in without_stopwords:
                neutral_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['neu'])
                positive_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['pos'])
                negative_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['neg'])
            for score in positive_sentences[:]:
                if score == 0.0:
                    positive_sentences.remove(score)
            for score in negative_sentences[:]:
                if score == 0.0:
                    negative_sentences.remove(score)
            for score in neutral_sentences[:]:
                if score == 0.0:
                    neutral_sentences.remove(score)
            POSITIVE = len(positive_sentences)
            NEGATIVE = len(negative_sentences)
            NEUTRAL = len(neutral_sentences)

            writer.writerow([entry, POSITIVE, NEGATIVE, NEUTRAL])
            print(positive_sentences)

            positive_sentences = []
            negative_sentences = []
            neutral_sentences = []



read_file = pd.read_csv(r'sentiment_analysis.csv')
read_file.to_excel(r'sentiment_whitepapers.xlsx', index=None, header=True)

            # overall_score = sum(compound_scores)/len(compound_scores)
            # if overall_score >= 0.05:
            #     print(overall_score, "Positive")
            #
            # elif overall_score <= - 0.05:
            #     print(overall_score, "Negative")
            #
            # else:
            #     print(overall_score, "Neutral")
    # returns the overall compound score of a white paper
    #https://github.com/cjhutto/vaderSentiment
    # positive sentiment : (compound score >= 0.05)
    # neutral sentiment : (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment : (compound score <= -0.05)




