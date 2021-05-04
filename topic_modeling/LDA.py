import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')
from PyPDF2 import PdfFileReader
import pyLDAvis
import re
import pandas as pd
from pprint import pprint
from tqdm import tqdm
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from tika import parser
# spacy for lemmatization
import spacy
nlp = spacy.load('en_core_web_sm-3.0.0')
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim import corpora, models

np.random.seed(2018)

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in ['miebach', 'whitepaper', 'white', 'paper']:
            result.append(lemmatize_stemming(token))
    return result


def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    listed_text = []
    text = ''

    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
        text = text.replace('\n', ' ')



    # rawText = parser.from_file(pdf)
    # clean_text = rawText['content'].replace('\n', '')
        n_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        g_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', n_sentences)  # adds whitespace after . and , 
        y_sentences = re.sub(r'\\xa0', r' ', g_sentences)  # changes \xa0 to whitespace 
        b_sentences = re.sub(r'% 20', r'', y_sentences)  # changes %20 to whitespace 
        t_best_sentences = re.sub(r'% 40', r'', b_sentences)  # changes %40 to whitespace 
        gr_sentences = re.sub(r'[-]', r'', t_best_sentences)
        grd_sentences = re.sub(r'Š', r' ', gr_sentences)
        grdf_sentences = re.sub(r'™', r"'", grd_sentences)
        fr_sentences = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', grdf_sentences)  # adds whitespace between number and letters 
        li = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", fr_sentences)  # remove space between link 
        li2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", li)
        clean = preprocess(li2)
        listed_text = clean


    return listed_text

def build_corpus_from_dir(dir_path):
    corpus = []
    brandnames_and_filenames = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):

        directory = root.split('/') #\\ for windows

        for name in filenames:
            f = os.path.join(root, name)
            brandname = directory[-1]
            filename = name
            brandnames_and_filenames.append([brandname, filename])


            if '.pdf' in name:
                pdf = PdfFileReader(f, "rb")
                # pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)

    return corpus, brandnames_and_filenames


if __name__ == '__main__':
    corpus, brandnames_and_filenames = build_corpus_from_dir(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms/Digital Technology')
    dictionary = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=2, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.sklearn.prepare(lda_model_tfidf, corpus, dictionary)
    # vis



