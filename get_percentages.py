import os
import pdftotext
from PyPDF2 import PdfFileReader
from tika import parser
import re
import csv
import os

phone_numbers = re.compile('[-+][0-9]+ +[0-9]+ +[0-9]+ +[0-9]+')
other_phone_numbers = re.compile('[+][0-9]+ +[0-9]+-[0-9]+-[0-9]')
more_phone_numbers = re.compile('[+][0-9]+ +\([0-9]\)[0-9]+ +[0-9]+ +[0-9]+')
more_phone_numbers2 = re.compile('[+][0-9]+ +\([0-9]\) +[0-9]+ +[0-9]+ +[0-9]+')

numbers = re.compile('[0-9]+')
phone_numbers_list = []
for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    for filename in files:
        print(filename)
        rawText = parser.from_file(root + '/' + filename)
        clean_text = rawText['content'].replace('\n', '')
        terms = clean_text.split(' ')
        number_list = numbers.findall(clean_text)
        phone_numbers_list.append(phone_numbers.findall(clean_text))
        phone_numbers_list.append(other_phone_numbers.findall(clean_text))
        phone_numbers_list.append(more_phone_numbers.findall(clean_text))
        phone_numbers_list.append(more_phone_numbers2.findall(clean_text))

        print(number_list)
        print(phone_numbers_list)
        # for n in number:
        #     res = clean_text.partition(str(n))[2]
        #     breakpoint()
        #     for term in res.split(' '):
        #         phonenumbers = phone_numbers.findall(res)
        #         phone_number.append(phonenumbers)
        #         if term.isalpha():
        #             res2 = clean_text.partition(str(term))[2]
        # numbers = [int(s) for s in clean_text.split() if s.isdigit()]
        # print(numbers)
        # for term in terms:
        #     if term.isdigit():
        #         print(term)
        # numbers = sum(c.isdigit() for c in terms)
        # print(numbers)
            # try:
            #     data = rawText['metadata']
            #     print(data)
            # except:
            #     continue



            # pdf_toread = PdfFileReader(open(root + "/" + filename, "rb"))
            # pdf_info = pdf_toread.getDocumentInfo()
            # AUTHORS = pdf_info.author
            # TITLE = pdf_info.title

