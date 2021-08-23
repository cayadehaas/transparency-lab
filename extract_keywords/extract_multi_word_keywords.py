#https://towardsdatascience.com/extract-keywords-from-documents-unsupervised-d6474ed38179
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re
import csv
import textract
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
ignored_words = list(stopwords.words('english'))
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


def pdf2text(pdf):
    """Iterate over pages and extract text"""
    try:
        txt = textract.process(pdf, method='pdfminer')
        text = str(txt)
        # Pre-processing
        t1 = text.replace('˛', 'fi')
        t2 = t1.replace('\n', '')
        t3 = t2.replace('˝', 'ffi')
        t4 = t3.replace('Š', ',')
        t5 = t4.replace('˜', 'ff')
        t6 = t5.replace('˚', 'fl')
        t7 = t6.replace('™', "'")
        t8 = t7.replace('ł', "")
        t9 = t8.replace('\\n', '')
        t10 = ''.join(map(lambda t9: t9.strip(), t9.split('-'))).split()  # removes unnecessary spaces
        t11 = ' '.join(t10)

        # add whitespace between components
        t12 = re.sub(r'([a-z])([A-Z])', r'\1 \2', t11)
        t13 = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', t12)
        t14 = re.sub(r'([a-z])([0-9])', r'\1 \2', t13)

        t15 = re.sub(r'(?<=[,?!%])(?=[^\s])', r' ', t14)
        t16 = re.sub(r'\\xa0', r' ', t15)
        t17 = re.sub(r'% 20', r'', t16)
        t18 = re.sub(r'% 40', r'', t17)
        t19 = re.sub(r'\\[a-z]+', r'', t18)
        t20 = re.sub(r'\\[a-z] [0-9]', r'', t19)
        t21 = re.sub(r'\\[a-z] [0-9] [a-z]', r'', t20)
        t22 = re.sub(r'[0-9][0-9]\\[a-z][a-z]', r'', t21)
        t23 = re.sub(r' [0-9]+\\x [0-9]+ ', r'', t22)
        t24 = re.sub(r'[0-9]\\[a-z]+', r'', t23)
        t25 = re.sub(r'[0-9]\\x [a-z]', r'', t24)
        t26 = re.sub(r'[-]', r'', t25)
        t27 = re.sub(r'[0-9] [0-9]+ [0-9]', r'', t26)
        t28 = re.sub(r'[0-9] [a-z]', r'', t27)
        t29 = re.sub(r"[a-z]\'", r'', t28)
        t30 = re.sub(r"/'b", r'', t29)
        t31 = re.sub(r'\.\.+', r'', t30)
        t32 = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", t31)
        t33 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", t32)

        # remove email addresses
        t34 = ' '.join(x for x in t33.split() if not x.startswith('http'))
        t35 = ' '.join(x for x in t34.split() if not x.startswith('www'))
        t36 = ' '.join(x for x in t35.split() if not x.endswith('com'))
        t37 = " ".join(x for x in t36.split() if '@' not in x)

        return t37
    except textract.exceptions.ShellError:  # when pdf cannot be opened
        return ""


def build_corpus_from_dir(dir_path):
    corpus = []
    brandnames_and_filenames = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        directory = root.split('/')
        for name in filenames:
            brandname = directory[-1]
            filename = name
            brandnames_and_filenames.append([brandname, filename])

            if '.pdf' in name:
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)
    return corpus, brandnames_and_filenames


if __name__ == '__main__':
    with open('/Users/cayadehaas/PycharmProjects/Transparency-Lab/extract_keywords/keywords_whitepapers.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["BRANDNAME", "FILENAME", "keywords"])

        corpus, brandnames_and_filenames = build_corpus_from_dir(
            r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdf_consultancy_firms/accenture')

        count_vec = CountVectorizer(
            ngram_range=(1, 1)
            , stop_words=ignored_words
        )
        text_set = [doc for doc in corpus]
        tf_result = count_vec.fit_transform(text_set)
        tf_result_df = pd.DataFrame(tf_result.toarray()
                                    , columns=count_vec.get_feature_names())
        the_sum_s = tf_result_df.sum(axis=0)
        the_sum_df = pd.DataFrame({
            'keyword': the_sum_s.index
            , 'tf_sum': the_sum_s.values
        })
        the_sum_df = the_sum_df[
            the_sum_df['tf_sum'] > 2
            ].sort_values(by=['tf_sum'], ascending=False)

        start_index = int(len(the_sum_df) * 0.01)
        my_word_df = the_sum_df.iloc[start_index:]
        my_word_df = my_word_df[my_word_df['keyword'].str.len() > 4]
        my_word_df = my_word_df[my_word_df['keyword'].str.isalpha()]

        # Bigram extraction
        text_set_biwords = [word_tokenize(doc) for doc in corpus]
        bigram_measures = BigramAssocMeasures()
        biword_finder = BigramCollocationFinder.from_documents(text_set_biwords)
        biword_finder.apply_freq_filter(3)
        biword_finder.apply_word_filter(lambda w:
                                        len(w) < 3
                                        or len(w) > 15
                                        or w.lower() in ignored_words)
        biword_phrase_result = biword_finder.nbest(bigram_measures.pmi, 20000)
        biword_colloc_strings = [w1 + ' ' + w2 for w1, w2 in biword_phrase_result]

        my_vocabulary = []
        my_vocabulary.extend(my_word_df['keyword'].tolist())
        my_vocabulary.extend(biword_colloc_strings)

        vec = TfidfVectorizer(
            analyzer='word'
            , ngram_range=(1, 2)
            , vocabulary=my_vocabulary)
        text_set = [doc for doc in corpus]
        tf_idf = vec.fit_transform(text_set)
        result_tfidf = pd.DataFrame(tf_idf.toarray()
                                    , columns=vec.get_feature_names())
        for index_brand, brand_file in enumerate(brandnames_and_filenames):
            for index, doc in enumerate(corpus):
                if index_brand == index:
                    brandname = brand_file[0]
                    filename = brand_file[1]

                    file_index = index_brand

                    test_tfidf_row = result_tfidf.loc[file_index]
                    keywords_df = pd.DataFrame({
                        'keyword': test_tfidf_row.index,
                        'tf-idf': test_tfidf_row.values
                    })
                    keywords_df = keywords_df[
                        keywords_df['tf-idf'] > 0
                        ].sort_values(by=['tf-idf'], ascending=False)

                    bigram_words = [item.split()
                                    for item in keywords_df['keyword'].tolist()
                                    if len(item.split()) == 2]
                    bigram_words_set = set(subitem
                                           for item in bigram_words
                                           for subitem in item)
                    keywords_df_new_biwords = keywords_df[~keywords_df['keyword'].isin(bigram_words_set)]
                    keywords = keywords_df_new_biwords['keyword'][:30]  # change number depending on amount of keywords wanted
                    keywords_30 = []
                    for keyword in keywords:
                        keywords_30.append(keyword)
                    print([brandname, filename, keywords_30])
                    writer.writerow([brandname, filename, keywords_30])

    read_file = pd.read_csv(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/extract_keywords/keywords_whitepapers.csv')
    read_file.to_excel(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/extract_keywords/keywords_whitepapers.xlsx', index=None, header=True)

