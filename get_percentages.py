import os
import pdftotext
from PyPDF2 import PdfFileReader
from tika import parser
import re
import csv
import os
import nltk
import spacy
n0 = ' [0-9]+ years'              #50 years
n1 = ' [1-2][0-9][0-9][0-9] '     #2020
n2 = '[0-9]+[+] percent'          #15+ percent
n3 = '[0-9]+ percent'             #15 percent
n4 = '[0-9]+\%'                   #97%
n5 = '[0-9]+\.[0-9]+\%'           #99.5%
n6 = '\$[0-9]+'                   #$12
n7 = '[0-9]+\.[0-9]+ +[a-z]+'     #1.1 billion
n8 = '[0-9]+\,[0-9]+ +[a-z]+'     #5,000 people
n9 = '[0-9]+\.[0-9]+ +[a-z]+'     #5.000 people
n10 = ' [0-9]+ +[a-z]+ '          #15 respondents
nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')
non_cardinal_words = ['percent', 'years']

with open('white_papers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FILENAME", "NUMBERS", "CARDINAL", "ORDINAL", "PERCENT"])
    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
        for filename in files:
            print(filename)
            rawText = parser.from_file(root + '/' + filename)
            clean_text = rawText['content'].replace('\n', '')
            terms = clean_text.split(' ')
            number_list = re.compile("(%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s)" % (n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)).findall(clean_text)
            cardinal_list = re.compile("(%s|%s|%s|%s)" % (n7, n8, n9, n10)).findall(clean_text)
            percent_list = re.compile("(%s|%s|%s|%s)" % (n2, n3, n4, n5)).findall(clean_text)
            for item in cardinal_list[:]:
                for word in non_cardinal_words:
                    if word in item:
                        cardinal_list.remove(item)
            NUMBERS = len(number_list)
            CARDINAL = len(cardinal_list)
            PERCENT = len(percent_list)

            writer.writerow([filename, NUMBERS, CARDINAL, "ORDINAL", PERCENT])

