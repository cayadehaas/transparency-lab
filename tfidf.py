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
tf_idf_values = []
for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    for filename in files:
        print(filename)
        english_stopwords = stopwords.words('english')
        rawText = parser.from_file(root + '/' + filename)
        clean_text = rawText['content'].replace('\n', '')
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
        breakpoint()
        df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
        new_df = df.sort_values(by=["tfidf"], ascending=False).head(50)

        breakpoint()
        # # df = pd.DataFrame(tf_idf_values)
        # # Create a Pandas Excel writer using XlsxWriter as the engine.
        # writer = pd.ExcelWriter('tf_idf_whitepapers.xlsx', engine='xlsxwriter')
        # # Convert the dataframe to an XlsxWriter Excel object.
        # new_df.to_excel(writer, sheet_name='Sheet1', index=False)
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.save()
