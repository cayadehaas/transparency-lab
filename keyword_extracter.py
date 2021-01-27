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
def get_text():
    list_of_lists = []
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/selectie_pdfs', topdown=False):
        directory = root.split('/')
        english_stopwords = stopwords.words('english')
        for entry in files:
            pdf = open('selectie_pdfs/' + entry, 'rb')
            rawText = parser.from_file('selectie_pdfs/' + entry)
            try:
                rawList = rawText['content'].splitlines()
                while '' in rawList: rawList.remove('')
                while ' ' in rawList: rawList.remove(' ')
                while '\t' in rawList: rawList.remove('\t')
                text = ''.join(rawList)
                text = text.lower()
                tokenized_text = nltk.word_tokenize(text)
            except:
                continue

            table = {ord(char): '' for char in string.punctuation}
            cleaned_messy_sentence = []
            for messy_word in tokenized_text:
                cleaned_word = messy_word.translate(table)
                cleaned_messy_sentence.append(cleaned_word)

            without_stopwords = []

            for token in cleaned_messy_sentence:
                if token not in english_stopwords:
                    without_stopwords.append(token)

            sentences = TreebankWordDetokenizer().detokenize(without_stopwords)
            list_of_lists.append(sentences)
    return entry, list_of_lists

    #         list_of_lists.append(without_stopwords)
    #
    # flat_list = [item for sublist in list_of_lists for item in sublist]
    # print(flat_list)


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def extract_keywords(flat_list):
    english_stopwords = stopwords.words('english')
    vectorizer = CountVectorizer(max_df=0.85, stop_words=english_stopwords, max_features=10000)
    X = vectorizer.fit_transform(flat_list)
    feature_names = list(vectorizer.vocabulary_.keys())
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)
    tf_idf_vector = tfidf_transformer.transform(vectorizer.transform(flat_list))
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, 100)
    for k in keywords:
        print(k, keywords[k])

def keywords_extracter(entry, flat_list):
    english_stopwords = stopwords.words('english')
    vectorizer = CountVectorizer(max_df=0.85, stop_words=english_stopwords, max_features=10000)
    X = vectorizer.fit_transform(flat_list)
    feature_names = list(vectorizer.vocabulary_.keys())
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(X)
    tf_idf_vector = tfidf_transformer.transform(vectorizer.transform(flat_list))

    results = []
    for i in range(tf_idf_vector.shape[0]):
        # get vector for a single document
        curr_vector = tf_idf_vector[i]

        # sort the tf-idf vector by descending order of scores
        sorted_items = sort_coo(curr_vector.tocoo())

        # extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items, 100)

        results.append(keywords)

        for k in keywords:
            print(k)

    df = pd.DataFrame(zip(flat_list, results), columns=['doc', 'keywords'])
    # Create a Pandas Excel writer using XlsxWriter as the engine.

    writer = pd.ExcelWriter('keywords.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():
    entry, flat_list = get_text()
    keywords_extracter(entry, flat_list)

    # extract_keywords(flat_list)


if __name__ == '__main__':
    main()


