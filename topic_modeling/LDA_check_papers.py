import PyPDF2
import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')
from PyPDF2 import utils
from PyPDF2 import PdfFileReader
import re
import csv
import pandas as pd
from pprint import pprint
import tqdm
from tqdm import tqdm
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from tika import parser
# spacy for lemmatization
import spacy
nlp = spacy.load('C:\Anaconda3\Lib\site-packages\en_core_web_sm/en_core_web_sm-3.0.0')
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel

np.random.seed(2018)

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return text
    # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in ['infosys', 'aarete', 'consultarc', 'abeam', 'accenture', 'acquisconsulting', 'aeqglobal', 'alacrita', 'alixpartner', 'altiusdata', 'altvil', 'altumconsulting', 'alvarezandmarsal', 'amaris', 'analysisgroup', 'ananyaaconsultingltd', 'ankura', 'appliedvaluegroup', 'appsbroker', 'astadia', 'auriemma', 'aurigaconsulting', 'avanade', 'avanceservices',
                                                                                                    'bain', 'bainbridgeconsulting', 'bakertilly', 'bateswhite', 'bdo', 'blayzegroup', 'bluematterconsulting', 'boozallen', 'bcg', 'bsienergyventures', 'boxtwenty', 'boxfusionconsulting', 'bpi', 'bridgespan', 'bts', 'bwbconsulting', 'capgemini', 'capitalone', 'cssfundraising', 'cellohealthbioconsulting', 'censeoconsulting', 'centricconsulting', 'crai',
                                                                                                    'clancy', 'clearedgeconsulting', 'clearviewconsulting', 'cleolifesciences', 'cobaltrecruitement', 'comsecglobal', 'concentra', 'consultingpoint', 'copehealthsolutions', 'cornerstone', 'corporate value', 'crowe', 'css-cg', 'curtins', 'dalberg', 'deallus', 'dean', 'dhg', 'dmt-group', 'doctors-management', 'eamesconsulting', 'eastwoordandpartners',
                                                                                                    'eca-uk', 'efficioconsulting', 'egonzehnder', 'elementaconsulting', 'elixirr', 'endeavormgmt' , 'enterey', 'establishinc', 'eunomia', 'everstgrp', 'everis', 'ey', 'fichtner', 'flattconsulting', 'fletcherspaght', 'flint-international', 'fminet', 'fsg', 'fticonsulting', 'ajg', 'gallup', 'gartner', 'gazellegc', 'gehealthcare', 'gep', 'glg', 'grantthornton',
                                                                                                    'grm-consulting', 'healthadvances', 'healthdimensionsgroup', 'hiltonconsulting', 'horvath-partners', 'huronconsultinggroup', 'i-pharmconsulting', 'ibm', 'ibuconsulting', 'infiniteglobal', 'infinityworks', 'infuse', 'innercircleconsulting', 'innogyconsulting', 'intellinet', 'iqvia', 'jabian', 'jdxconsulting', 'atkearney', 'klifesciences', 'kirtanaconsulting', 'knowledgecapitalgroup', 'kovaion',
                                                                                                    'kpmg', 'lda-design', 'lek', 'lexamed', 'lscconnect', 'magenic', 'marakon', 'marsandco', 'mmc', 'mcchrystalgroup', 'mckinsey', 'medcgroup', 'mercer', 'methodllp', 'bradbrookconsulting', 'mmgmc', 'mlmgroup', 'mottmac', 'navigant', 'neueda', 'newtoneurope', 'nmg-group', 'ndy', 'novantas', 'ocs-consulting', 'occstrategy', 'olivehorse', 'oliverwyman', 'opusthree', 'oxera', 'oxfordcorp', 'paconsulting', 'pmcretail', 'pearlmeyer', 'puralstrategy', 'pointb',
                                                                                                    'portasconsulting', 'pragmauk', 'prmaconsulting', 'prophet', 'protiviti', 'providge', 'ptsconsulting', 'publicconsultinggroup', 'putassoc', 'pwc', 'qa', 'red-badger', 'rhrinternationalconsultants', 'rise', 'robertwest', 'rsmus', 'rsp', 'russelreynolds', 'sia-partners', 'slalom', 'ssandco', 'argentwm', 'starlizard', 'sterlingassociates', 'sternstewart', 'londonstrategicconsulting', 'stroudinternational', 'syneoshealth',
                                                                                                    'synpulse', 'syntegragroup', 'talan', 'team-consulting', 'teamwill-consulting', 'tefen', 'idoc', 'chartis', 'eckrothplanning', 'thoughtprovokingconsulting', 'transformconsultinggroup', 'trinity-partners', 'ipeglobal', 'turnkeyconsulting', 'unboxed', 'vantagepartners', 'ivaldigroup', 'izientinc', 'miebach', 'whitepaper', 'white', 'paper', 'deloitte', 'cognizant', 'capgemini', 'gartner', 'wyman', 'oliver']:
            result.append(lemmatize_stemming(token))
    return result


def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    listed_text = []
    text = ''

    try:
        for i in range(pdf.getNumPages()):
            page = pdf.getPage(i)
            text = text + page.extractText()
            text = text.replace('\n', ' ')
            text = text.lower()

            n_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            g_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', n_sentences)  # adds whitespace after . and , 
            y_sentences = re.sub(r'\\xa0', r' ', g_sentences)  # changes \xa0 to whitespace 
            b_sentences = re.sub(r'% 20', r'', y_sentences)  # changes %20 to whitespace 
            t_best_sentences = re.sub(r'% 40', r'', b_sentences)  # changes %40 to whitespace 
            gr_sentences = re.sub(r'[-]', r'', t_best_sentences)
            grd_sentences = re.sub(r'Š', r' ', gr_sentences)
            grdf_sentences = re.sub(r'™', r"'", grd_sentences)
            fr_sentences = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2',
                                  grdf_sentences)  # adds whitespace between number and letters 
            li = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", fr_sentences)  # remove space between link 
            li2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", li)
            clean = preprocess(li2)
            listed_text = clean

    except utils.PdfReadError:
        listed_text = []


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

def make_bigrams_doc(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod[texts]

def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization_doc(texts, allowed_postags=None):
    """https://spacy.io/api/annotation"""
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']


    doc = nlp(" ".join(texts))
    texts_out = ([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def lemmatization(texts, allowed_postags=None):
    """https://spacy.io/api/annotation"""
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_lda_model(corpus, dictionary, k,a,b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    pprint(lda_model.print_topics())
    return lda_model

def compute_coherence_values(lda_model, texts, dictionary):

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


    return coherence_model_lda.get_coherence()

if __name__ == '__main__':
    corpus, brandnames_and_filenames = build_corpus_from_dir(r'F:\pdf_consultancy_firms\CLASSIFIED_PAPERS/Agile')
    breakpoint()
    # corpus = [x for x in corpus if x != []] #remove empty lists

    #Bigrams, comment out if you're using trigrams
    data_words_bigrams = make_bigrams(corpus)
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #Trigrams, comment out if you're using bigrams
    # data_words_trigrams = make_trigrams(corpus)
    # data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    texts = data_lemmatized
    dictionary = gensim.corpora.Dictionary(data_lemmatized)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]

    # tfidf = models.TfidfModel(bow_corpus)
    # corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=7, id2word=dictionary, alpha='symmetric', eta=0.9099999999999999, passes=2, workers=4)
    # pprint(lda_model.print_topics(num_words=50))
    breakpoint()
    for i in range(len(bow_corpus)):
        print(lda_model[bow_corpus[i]])
    # with open('topic_modeling_LDA_bigrams_250_keywords.csv', 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Topic", "Word"])
    #     for idx, topic in lda_model.print_topics(-1, num_words=50):
    #         print('Topic: {} Word: {}'.format(idx, topic))
    #         writer.writerow([idx, topic])
    # for root, dirs, files in os.walk('F:\pdf_consultancy_firms\CLASSIFIED_PAPERS/Agile',
    #                                  topdown=False):
    #     for filename in tqdm(files):
    #         if ".pdf" in filename:
    #             text = ''
    #             print(filename)
    #             f = os.path.join(root, filename)
    #             pdf = PdfFileReader(f, "rb")
    #             document = pdf2text(pdf)
    #             data_words_bigrams = make_bigrams_doc(document)
    #             data_lemmatized = lemmatization_doc(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #             texts = data_lemmatized
    #             texts = ' '.join(texts)
    #
    #             bow_vector = dictionary.doc2bow(preprocess(texts))
    #             for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    #                 print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
