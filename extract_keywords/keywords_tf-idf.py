#https://www.holisticseo.digital/python-seo/categorize-queries/
import os
from tika import parser
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
from nltk.corpus import stopwords
import nltk
import string
import re
from tqdm import tqdm
tf_idf_values = []
keywords = []
import csv
with open('keywords_white_papers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FILENAME", "keywords"])
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms', topdown=False):
        for filename in tqdm(files):
            if ".pdf" in filename:
                print(filename)
                english_stopwords = stopwords.words('english')
                rawText = parser.from_file(root + '/' + filename)
                clean_text = rawText['content'].replace('\n', '')
                rawList = rawText['content'].splitlines()
                while '' in rawList: rawList.remove('')
                while ' ' in rawList: rawList.remove(' ')
                while '\t' in rawList: rawList.remove('\t')
                text = ''.join(rawList)
                text = text.lower()
                new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                                       text)  # adds whitespace after between small and captial letter
                good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', new_sentences)  # adds whitespace after . and ,
                normal_sentences = re.sub(r'\\xa0', r' ', good_sentences)  # changes \xa0 to whitespace
                best_sentences = re.sub(r'% 20', r'', normal_sentences)  # changes %20 to whitespace
                the_best_sentences = re.sub(r'% 40', r'', best_sentences)  # changes %40 to whitespace
                great_sentences = re.sub(r'[-]', r'', the_best_sentences)
                free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2',
                                        great_sentences)  # adds whitespace between number and letters
                links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", free_sentences)  # remove space between link
                links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link

                tokenized_text = nltk.word_tokenize(links2)

                nltk.pos_tag(tokenized_text)
                words = []
                for word, tag in nltk.pos_tag(tokenized_text):
                    #NN noun, singular,  NNS noun plural, NNP proper noun singular , NNPS proper noun plural
                    if tag in ['NN', 'NNS',  'NNP' ,'NNPS']:
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
                tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform([text_without_stopwords])
                first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[-1]
                df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
                new_df = df.sort_values(by=["tfidf"], ascending=False).head(20)
                for keyword in new_df.index:
                    keywords.append(keyword)
                keyword_list = ", ".join(keywords)
                writer.writerow([filename, keyword_list])
                keywords = []

read_file = pd.read_csv(r'keywords_white_papers.csv')
read_file.to_excel(r'keywords_white_papers.xlsx', index=None, header=True)


