import pytesseract as pytesseract
from PyPDF2 import PdfFileReader
from matplotlib.pyplot import stem
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/pytesseract'


def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    images = convert_from_path(pdf)
    NBR_OF_PAGES = len(images)
    complete_PDF_List = []
    page = 0
    text = ''
    for i in range(len(images)):
        print(f'Working on File:{pdf} Page No: {page + 1}')
        extractedInformation = pytesseract.image_to_string(images[i])
        extractedInformation = extractedInformation.replace('\n', '')
        n_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', extractedInformation)
        g_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', n_sentences)  # adds whitespace after . and , 
        y_sentences = re.sub(r'\\xa0', r' ', g_sentences)  # changes \xa0 to whitespace 
        b_sentences = re.sub(r'% 20', r'', y_sentences)  # changes %20 to whitespace 
        t_best_sentences = re.sub(r'% 40', r'', b_sentences)  # changes %40 to whitespace 
        gr_sentences = re.sub(r'[-]', r'', t_best_sentences)
        fr_sentences = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', gr_sentences)  # adds whitespace between number and letters 
        li = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", fr_sentences)  # remove space between link 
        li2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", li)
        text += li2
        page + 1

    return text
    # text = ''
    # for i in range(pdf.getNumPages()):
    #     page = pdf.getPage(i)
    #     text = text + page.extractText()
    # return text

def stem_tokenize(document):
    '''return stemmed words longer than 2 chars and all alpha'''
    tokens = [stem(w) for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens

def tokenize(document):
    '''return words longer than 2 chars and all alpha'''
    tokens = [w for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens

def build_corpus_from_dir(dir_path):
    corpus = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        for name in filenames:
            f = os.path.join(root, name)
            if '.pdf' in name:
                # pdf = PdfFileReader(f, "rb")
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus


if __name__ == '__main__':
    corpus = build_corpus_from_dir(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms/WHITEPAPERS')
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop)
    vectorizer.fit(corpus)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    for doc in corpus:
        breakpoint()
        doc = ' '.join(doc.split())
        # print(' '.join(doc.split())[500])
        tdm = vectorizer.transform([doc])
        dense = tdm.todense()
        episode = dense[0].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
        sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
        for phrase, score in [(features[word_id], score) for (word_id, score) in sorted_phrase_scores][:10]:
           print('{0: <20} {1}'.format(phrase, score))
        print()