import os
import pdftotext
from PyPDF2 import PdfFileReader
from tika import parser
import re
import csv
import os
import nltk
import spacy
import pandas as pd
import string

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

nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')

with open('transitions_text_structure.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FILENAME", "#ILLUSTRATIONS", "#ADDITIONS", "#CONTRAST", "#COMPARISON"])
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
        for filename in files:
            print(filename)
            rawText = parser.from_file(root + '/' + filename)
            clean_text = rawText['content'].replace('\n', '')
            sentences_with_space_after_dot_comma = re.sub(r'(?<=[.,])(?=[^\s])', r' ', clean_text)

            sentences = nltk.sent_tokenize(sentences_with_space_after_dot_comma)
            good_sentences = re.sub(r'[-]', r'', sentences_with_space_after_dot_comma)

            transition_list = re.compile(
                "(%s|%s|%s|%s)" % (n11, n12, n13, n14)).findall(good_sentences)
            illustriation_list = re.compile(
                "(%s)" % (n11)).findall(clean_text)
            addition_list = re.compile(
                "(%s)" % (n12)).findall(clean_text)
            contrast_list = re.compile(
                "(%s)" % (n13)).findall(clean_text)
            comparison_list = re.compile(
                "(%s)" % (n14)).findall(clean_text)
            ILLUSTRATIONS = len(illustriation_list)
            ADDITIONS = len(addition_list)
            CONTRAST = len(contrast_list)
            COMPARISON = len(comparison_list)
            print("ILLUSTRATIONS:", ILLUSTRATIONS, "ADDITIONS:", ADDITIONS, "CONTRAST:", CONTRAST, "COMPARISON:", COMPARISON)

            writer.writerow([filename, ILLUSTRATIONS, ADDITIONS, CONTRAST, COMPARISON])

read_file = pd.read_csv(r'transitions_text_structure.csv')
read_file.to_excel(r'transitions_text_structure.xlsx', index=None, header=True)



