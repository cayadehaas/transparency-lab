from __future__ import print_function

import pytesseract as pytesseract
from PyPDF2 import PdfFileReader
from matplotlib.pyplot import stem
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
import re
import csv
from pdf2image import convert_from_path
from tqdm import tqdm
from tika import parser
import string
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import joblib
from sklearn.cluster import KMeans

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
            result.append(token)

    return result

def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    rawText = parser.from_file(pdf)
    clean_text = rawText['content'].replace('\n', '')
    rawList = rawText['content'].splitlines()
    while '' in rawList: rawList.remove('')
    while ' ' in rawList: rawList.remove(' ')
    while '\t' in rawList: rawList.remove('\t')
    text = ' '.join(rawList)
    text = text.lower()
    new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                           text)  # adds whitespace after between small and captial letter
    good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', new_sentences)  # adds whitespace after . and ,
    normal_sentences = re.sub(r'\\xa0', r' ', good_sentences)  # changes \xa0 to whitespace
    best_sentences = re.sub(r'% 20', r'', normal_sentences)  # changes %20 to whitespace
    the_best_sentences = re.sub(r'% 40', r'', best_sentences)  # changes %40 to whitespace
    # great_sentences = re.sub(r'[-]', r'', the_best_sentences)
    free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2',
                            the_best_sentences)  # adds whitespace between number and letters
    links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", free_sentences)  # remove space between link
    links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link
    links3 = preprocess(links2)
    # tokenized_text = nltk.word_tokenize(links3)
    nltk.pos_tag(links3)
    words = []
    for word, tag in nltk.pos_tag(links3):
        # NN noun, singular,  NNS noun plural, NNP proper noun singular , NNPS proper noun plural
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            words.append(word)
    table = {ord(char): '' for char in string.punctuation}
    cleaned_messy_sentence = []
    for messy_word in words:
        cleaned_word = messy_word.translate(table)
        cleaned_messy_sentence.append(cleaned_word)

    without_stopwords = []
    for token in cleaned_messy_sentence:
        if token not in english_stopwords:
            without_stopwords.append(token)
    text_without_stopwords = ' '.join(without_stopwords)
    return text_without_stopwords


def stem_tokenize(document):
    '''return stemmed words longer than 2 chars and all alpha'''
    tokens = [stem(w) for w in document.split() if len(w) > 3 and w.isalpha()]
    return tokens


def tokenize(document):
    '''return words longer than 2 chars and all alpha'''
    tokens = [w for w in document.split() if len(w) > 3 and w.isalpha()]
    return tokens


def build_corpus_from_dir(dir_path):

    corpus = []
    brandnames_and_filenames = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        directory = root.split('/')
        for name in filenames:
            f = os.path.join(root, name)
            # topic = directory[-2]
            brandname = directory[-1]
            # filename = name
            brandnames_and_filenames.append([brandname]) #removed filename for clear plot

            if '.pdf' in name:
                # pdf = PdfFileReader(f, "rb")
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus, brandnames_and_filenames


if __name__ == '__main__':
    corpus, brandnames_and_filenames = build_corpus_from_dir(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/topic_modeling/Agile')
    vectorizer = TfidfVectorizer(stop_words=english_stopwords, tokenizer=tokenize)
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tf_idf_matrix)

    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tf_idf_matrix)
    clusters = km.labels_.tolist()
    Agile_pdfs = {'title': brandnames_and_filenames, 'corpus': corpus, 'cluster': clusters, }
    frame = pd.DataFrame(Agile_pdfs, index=[clusters], columns=['title', 'corpus', 'cluster'])
    print(frame)
    # # print(len(clusters))
    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    # km = joblib.load('doc_cluster.pkl')
    # clusters = km.labels_.tolist()

    # # create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=brandnames_and_filenames))
    # group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    plt.title("LSA Document Similarity: Agile")
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,

                color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the title
    for i in range(len(df)):
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

    plt.show()  # show the plot

    # uncomment the below to save the plot if need be
    plt.savefig('clusters_agile_papers.png', dpi=200)

    # for index_brand, brand_file in enumerate(brandnames_and_filenames):
    #     for index, doc in enumerate(corpus):
    #         if index_brand == index:
    #             topic = brand_file[0]
    #             brandname = brand_file[1]
    #             filename = brand_file[2]
    #             keywords = []
    #             tf_idf_scores = []
    #             scores = []
    #             doc = ' '.join(doc.split())
    #
    #
    #             tdm = vectorizer.transform([doc])
    #             dense = tdm.todense()
    #             episode = dense[0].tolist()[0]
    #             phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    #             sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    #             for phrase, score in [(features[word_id], score) for (word_id, score) in sorted_phrase_scores][:50]:
    #                 keywords.append(phrase)
    #                 tf_idf_scores.append('{0} {1}'.format(phrase, score))
    #                 scores.append(score)
    #                 print('{0: <20} {1}'.format(phrase, score))
    #             print()
    #             tf_idf_scores_list = ', '.join(tf_idf_scores)
    #             keywords_list = ', '.join(keywords)
    #             average_tf_idf_score = sum(scores) / len(scores)

