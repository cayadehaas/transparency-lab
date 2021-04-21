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
stop = stopwords.words('english')
import re
import csv
from pdf2image import convert_from_path
from tqdm import tqdm
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


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
        page += 1

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
    brandnames_and_filenames = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        directory = root.split('\\')
        for name in filenames:
            f = os.path.join(root, name)
            topic = directory[-2]
            brandname = directory[-1]
            filename = name
            brandnames_and_filenames.append([topic, brandname, filename])


            if '.pdf' in name:
                # pdf = PdfFileReader(f, "rb")
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus, brandnames_and_filenames

n11 = '[tT]hus|[fF]or example|[fF]or instance|[nNa]mely|[tT]o illustrate|[iI]n other words|[iI]n particular|' \
      '[sS]pecifically|[sS]uch as|[tT]o demonstrate'   #Illustration
n12 = '[aA]nd|[iI]n addition to|[fF]urthermore|[mM]oreover|[bB]esides|[tT]han|[tT]oo|[aA]lso|[Bb]oth|[Aa]nother| '\
      '[eE]qually important|[Aa]gain|[Ff]urther|[Ff]inally|[Nn]ot only|[Aa]s well as|[Ii]n the second place|[Nn]ext|' \
      '[Ll]ikewise|[Ss]imilarly|[Ii]n fact|[Aa]s a result|[Cc]onsequently|[Ii]n the same way|' \
      '[Tt]herefore|[Oo]therwise|[fF]irst|[sS]econd|[tT]hird|[fF]ourth|[lL]ast'   #Addition
n13 = '[Oo]n the contrary|[Cc]ontrarily|[nN]otwithstanding|[Bb]ut|[hH]owever|[Nn]evertheless|[iI]n spite of|' \
      '[iI]n contrast|[Yy]et|[Oo]n one hand|[Oo]n the other hand|[Rr]ather|[Cc]onversely|[Aat] the same time|' \
      '[Ww]hile this may be true' #Contrast
n14 = '[iI]n the same way|[bB]y the same token|[Ss]imilarly|[Ii]n like manner|[Ll]ikewise|[Ii]n similar fashion' #Comparision


if __name__ == '__main__':
    with open('../matrix_whitepapers.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # writer.writerow(
        #     ["TOPIC", "BRANDNAME", "FILENAME", "#ADDITIONS", "#CONTRASTS", "#COMPARISIONS",
        #      "AVERAGE TF-IDF SCORE", "KEYWORDS", "KEYWORDS WITH TF-IDF SCORE"])

        corpus, brandnames_and_filenames = build_corpus_from_dir(r'F:\consultancy_firms_pdfs\CLASSIFIED_WHITEPAPERS\Commerce - Marketing')
        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop)
        vectorizer.fit(corpus)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()

        for index_brand, brand_file in enumerate(brandnames_and_filenames):
            for index, doc in enumerate(corpus):
                if index_brand == index:
                    topic = brand_file[0]
                    brandname = brand_file[1]
                    filename = brand_file[2]
                    keywords = []
                    tf_idf_scores = []
                    scores = []
                    doc = ' '.join(doc.split())

                    # transition_list = re.compile(
                    #     "(%s|%s|%s|%s)" % (n11, n12, n13, n14)).findall(doc)
                    # illustriation_list = re.compile(
                    #     "(%s)" % (n11)).findall(doc)
                    addition_list = re.compile(
                        "(%s)" % (n12)).findall(doc)
                    contrast_list = re.compile(
                        "(%s)" % (n13)).findall(doc)
                    comparison_list = re.compile(
                        "(%s)" % (n14)).findall(doc)
                    # ILLUSTRATIONS = len(illustriation_list)
                    ADDITIONS = len(addition_list)
                    CONTRAST = len(contrast_list)
                    COMPARISON = len(comparison_list)
                    print("ADDITIONS:", ADDITIONS, "CONTRAST:", CONTRAST,
                          "COMPARISON:", COMPARISON)

                    tdm = vectorizer.transform([doc])
                    dense = tdm.todense()
                    episode = dense[0].tolist()[0]
                    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
                    for phrase, score in [(features[word_id], score) for (word_id, score) in sorted_phrase_scores][:50]:
                        keywords.append(phrase)
                        tf_idf_scores.append('{0} {1}'.format(phrase, score))
                        scores.append(score)
                        print('{0: <20} {1}'.format(phrase, score))
                    print()
                    tf_idf_scores_list = ', '.join(tf_idf_scores)
                    keywords_list = ', '.join(keywords)
                    average_tf_idf_score = sum(scores) / len(scores)

                    print([topic, brandname, filename, ADDITIONS, CONTRAST, COMPARISON, average_tf_idf_score, keywords_list, tf_idf_scores_list])
                    writer.writerow([topic, brandname, filename, ADDITIONS, CONTRAST, COMPARISON, average_tf_idf_score, keywords_list, tf_idf_scores_list])

    read_file = pd.read_csv(r'../matrix_whitepapers.csv')
    read_file.to_excel(r'matrix_whitepapers.xlsx', index=None, header=True)


