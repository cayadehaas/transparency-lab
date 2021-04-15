import os
from PyPDF2 import PdfFileReader
from tika import parser
import re
import nltk
import csv
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_model = SentimentIntensityAnalyzer()
import spacy
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
import csv
from nltk.tokenize import RegexpTokenizer
tf_idf_values = []
keywords = []
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')
nlp = spacy.load("en_core_web_sm")
sentences_on_page = []
def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=set(),
              verbose=0):
    """
    Run VADER on a sentence from spacy

    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -empty set -> all parts of speech are provided
    -non-empty set: only these parts of speech are considered
    :param int verbose: if set to 1, information is printed
    about input and output

    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        pass
        # print()
        # print('INPUT SENTENCE', sent)
        # print('INPUT TO VADER', input_to_vader)
        # print('VADER OUTPUT', scores)

    return scores

compound_sentences = []
verhaallijn = []

n1 = '[0-9]+ years'              #50 years
n2 = ' [1-2][0-9][0-9][0-9] '     #2020
n7 = '[0-9]+\.[0-9]+ +[a-z]+'  # 1.1 billion
n8 = '[0-9]+\,[0-9]+ +[a-z]+'  # 5,000 people
n9 = '[0-9]+\.[0-9]+ +[a-z]+'  # 5.000 people
n10 = '[0-9]+ +[a-z]+'  # 15 respondents
n11 = '\+1 [0-9]+ +[0-9]+ +[0-9]+ +[a-z]+' #+1 630 258 2402 w+
n12 = '\+[0-9]+ [0-9]+-[0-9]+-[0-9] +[a-z]+'#+49 69-273992-0 w+
n13 = '\+[0-9]+ [0-9]+ [0-9]+ [0-9]+ +[a-z]+' #+44 370 904 3601 w+
n14 = '\+[0-9]+ +[0-9]+ [a-z]+' #+49 692739920 w+
n15 = '\+ [0-9]+ +[0-9]+ +[0-9]+ +[a-z]+' #+ 49 69 2739920 w+
n16 = '[0-9]+ percent' #45 percent
n17 = '[0-9]+%' #45%
n18 = '[1-2][0-9][0-9][0-9] ' #2018

image_counter = 0
with open('../information_per_page.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ["BRANDNAME", "FILENAME", 'PAGE', 'NBR. OF NEGATIVE SENTENCES', 'NBR. OF NEUTRAL SENTENCES',
         'NBR. OF POSITIVE SENTENCES', 'AMOUNT OF NUMBERS', 'AMOUNT OF QUESTION MARKS'])
    for root, dirs, files in os.walk(r'F:\consultancy_firms_pdfs\CLASSIFIED_WHITEPAPERS', topdown=False):
        directory = root.split('/')
        for file in tqdm(files):
            firm = directory[-1]
            #use tika parser to clearly get sentences
            #use pdffilereader to extract text from certain page

            # try:
            #     pdf_toread = PdfFileReader(open(root + "/" + file, "rb"))
            #     pdf_info = pdf_toread.getDocumentInfo()
            #
            #     for page_nr in range(pdf_toread.numPages):  # iterate over the pages
            #         print('Page', page_nr+1)
            #         page = pdf_toread.getPage(page_nr)
            #         page0 = pdf_toread.getPage(page_nr)
            #         if '/XObject' in page0['/Resources']:
            #             xObject = page0['/Resources']['/XObject'].getObject()
            #             for obj in xObject:
            #                 # if xObject[obj]['/Subtype'] == '/Image':
            #                 #     image_counter += 1
            #                 if xObject[obj]['/Subtype'] == '/Form':
            #                     image_counter += 1
            #         AMOUNT_OF_IMAGES = image_counter
            #         image_counter = 0
            #         page_nbr = page_nr + 1
            pdf_file = root + '/' + file
            try:
                images = convert_from_path(pdf_file)
                NBR_OF_PAGES = len(images)
                complete_PDF_List = []
                page = 0
                for i in range(len(images)):
                    print(f'Working on File:{file} Page No: {page + 1}')
                    text = []
                    extractedInformation = pytesseract.image_to_string(images[i])
                    extractedInformation = extractedInformation.replace('\n', '')

                    n_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', extractedInformation)
                    g_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', n_sentences)  # adds whitespace after . and ,
                    y_sentences = re.sub(r'\\xa0', r' ', g_sentences)  # changes \xa0 to whitespace
                    b_sentences = re.sub(r'% 20', r'', y_sentences)  # changes %20 to whitespace
                    t_best_sentences = re.sub(r'% 40', r'', b_sentences)  # changes %40 to whitespace
                    gr_sentences = re.sub(r'[-]', r'', t_best_sentences)
                    fr_sentences = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2',
                                          gr_sentences)  # adds whitespace between number and letters
                    li = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", fr_sentences)  # remove space between link
                    li2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", li)

                    # als amount getallen groter is dan bepaald aantal skip de pagina
                    # if sum(c.isdigit() for c in li2) > 100:
                    #     continue
                    # matches = ['references', 'about the author', 'citations']
                    # if any(x in li2.lower() for x in matches):
                    #     break
                    # intro = ['table of contents', 'list of figures']
                    # if any(x in li2.lower() for x in intro):
                    #     continue

                    vraagtekens = re.compile('[?]').findall(li2)
                    QUESTION_MARKS = len(vraagtekens)
                    # if len(vraagtekens) > 35:
                    #     break
                    # punct = ['!', '"', '#', '$', '&', '(', ')', '*', '+', ','
                    #     , '-', '/', ':', ';', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
                    # for c in punct:
                    #     li2 = li2.replace(c, "")
                    # li2 = " ".join(li2.split())  # remove whitespace

                    li22 = nltk.sent_tokenize(li2)

                    for sentence in li22:
                        numbers = ['percent ', '%']
                        compound_sentences.append(run_vader(sentence, lemmatize=True, verbose=1)['compound'])
                        # check for numbers, percent or % voorkomt in sentence, add to list
                        number_list = re.compile("(%s|%s|%s|%s|%s|%s)" % (n7, n8, n9, n10, n16, n17)).findall(str(li22))
                        years = re.compile("(%s|%s|%s)" % (n1, n2, n18)).findall(str(li22))
                        phone_numbers = re.compile("(%s|%s|%s|%s|%s)" % (n11, n12, n13, n14, n15)).findall(str(li22))

                        for year in years[:]:  # remove whitespace
                            new_year = year.strip()
                            years.remove(year)
                            years.append(new_year)

                        for number in phone_numbers[:]:
                            new_number = number.strip('-')
                            # phone_numbers.remove(number)
                            phone_numbers.append(new_number)

                        for number in number_list:  # remove years from number list
                            for element in years:
                                if element in number:
                                    try:
                                        number_list.remove(number)
                                    except:
                                        continue

                        for number in phone_numbers:  # remove phone numbers from number list
                            for element in number_list[:]:
                                if element in number:
                                    number_list.remove(element)

                    try:
                        NUMBERS = len(number_list)
                    except:
                        NUMBERS = 0

                    for score in compound_sentences:
                        if score >= 0.05:  # positive
                            verhaallijn.append(3)
                        elif score <= - 0.05:  # negative
                            verhaallijn.append(1)
                        else:  # neutral
                            verhaallijn.append(2)
                    #
                    # verhaallijn = (",".join(repr(e) for e in verhaallijn))
                    negative_sentences = 0
                    positive_sentences = 0
                    neutral_sentences = 0
                    for number in verhaallijn:
                        if number == 1:
                            negative_sentences += 1
                        if number == 2:
                            neutral_sentences += 1
                        if number == 3:
                            positive_sentences += 1
                    page += 1
                    writer.writerow(
                        [firm, file, 'BLZ.%s' % page, negative_sentences, neutral_sentences, positive_sentences, NUMBERS,
                         QUESTION_MARKS])
                    compound_sentences = []  # make list empty for next file
                    verhaallijn = []  # make list empty for next file]
            except:
                writer.writerow(
                    [firm, file, 'UNABLE TO OPEN WITH PDF FILE READER PACKAGE'])
                continue


read_file = pd.read_csv(r'../information_per_page.csv')
read_file.to_excel(r'information_per_page.xlsx', index=None, header=True)
