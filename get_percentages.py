import os
import pdftotext
from PyPDF2 import PdfFileReader
from tika import parser
import re
import csv
import os
#
# phone_nr_1 = r'[+][0-9]+ +[0-9]+ +[0-9]+ +[0-9]+'           #+1 630 258 2402
# phone_nr_2 = '[+][0-9]+ +[0-9]+-[0-9]+-[0-9]'               #+49 69-273992-0
# phone_nr_3 = '[+][0-9]+ +\([0-9]\)[0-9]+ +[0-9]+ +[0-9]+'   #+41 (0)22 786 2744
# phone_nr_4 = '[+][0-9]+ +\([0-9]\) +[0-9]+ +[0-9]+ +[0-9]+' #+41 (0) 22 869 1212
# phone_nr_5 = '[+] +[0-9]+\.[0-9]+\.[0-9]+'                  #+ 713.877.8130
# phone_nr_6 = '[+][0-9]+ +[0-9]+ +[0-9]+-[0-9]+'              #+49 69 273992­0

n1 = ' [0-9]+ years'              #50 years
n2 = ' [1-2][0-9][0-9][0-9] '     #2020
n3 = '[0-9]+ percent'             #15 percent
n4 = '[0-9]+\%'                   #97%
n5 = '[0-9]+\.[0-9]+\%'           #99.5%
n6 = '\$[0-9]+'                   #$12
n7 = '[0-9]+\.[0-9]+ +[a-z]+'     #1.1 billion
n8 = '[0-9]+\,[0-9]+ +[a-z]+'     #5,000 people
n9 = '[0-9]+\.[0-9]+ +[a-z]+'     #5.000 people
n10 = ' [0-9]+ +[a-z]+ '        #15 respondents

# phone_numbers_list = []
for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    for filename in files:
        print(filename)
        rawText = parser.from_file(root + '/' + filename)
        clean_text = rawText['content'].replace('\n', '')
        terms = clean_text.split(' ')
        number_list = re.compile("(%s|%s|%s|%s|%s|%s|%s|%s|%s|%s)" % (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)).findall(clean_text)
        NUMBERS = len(number_list)
        # phone_numbers_list = re.compile("(%s|%s|%s|%s)" % (phone_nr_1, phone_nr_2, phone_nr_3, phone_nr_4)).findall(clean_text)
        # phone_numbers_list.append(phone_numbers.findall(clean_text))
        # phone_numbers_list.append(other_phone_numbers.findall(clean_text))
        # phone_numbers_list.append(more_phone_numbers.findall(clean_text))
        # phone_numbers_list.append(more_phone_numbers2.findall(clean_text))

        print(number_list)
        print(NUMBERS)
        # print(phone_numbers_list)

        # for number in phone_numbers_list:
        #     new_set = {clean_text.replace(number, '') for x in clean_text}
        #
        # number_list = numbers.findall(new_set)

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

