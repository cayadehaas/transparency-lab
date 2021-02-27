import csv
from tika import parser
import textract, re
import pdftotext
import pandas as pd
import os
from PyPDF2 import PdfFileReader
import pdftitle
import glob
import re
# brandname, url, title , authors, nbr of pages, nbr of words, nbr of illustrations, nbr of graphs, nbr of numbers/percentages

with open('white_papers.csv', 'w', newline='') as file:
    image_counter = 0
    writer = csv.writer(file)
    writer.writerow(["BRANDNAME", "URL", "FILENAME", "TITLE", "AUTHORS", "EMAIL", "NBR. OF PAGES", "NBR. OF CHARACTERS", "USE OF IMAGES", "NBR. OF IMAGES", "NBR. OF NUMBERS"])

    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms/WHITEPAPERS/',
                                 topdown=False):
        directory = root.split('/')
        f = open("firms_and_url.txt", "r")
        firms = f.readlines()
        for firm in firms:
            for entry in files:
                print(entry, directory)
                firm_name = firm.split(',')
                if '.pdf' in entry:
                    with open(root + "/" + entry, "rb") as f:
                        FILENAME = entry
                        pdf = pdftotext.PDF(f)
                    nr_of_pages = len(pdf)

                    pdf_toread = PdfFileReader(open(root + "/" + entry, "rb"))
                    pdf_info = pdf_toread.getDocumentInfo()
                    AUTHORS = pdf_info.author
                    TITLE = pdf_info.title
                    for i in range(pdf_toread.numPages - 1):
                        page0 = pdf_toread.getPage(i + 1)
                        if '/XObject' in page0['/Resources']:
                            xObject = page0['/Resources']['/XObject'].getObject()
                            for obj in xObject:
                                if xObject[obj]['/Subtype'] == '/Image':
                                    use_of_images = 'YES'
                                    image_counter += 1
                                if xObject[obj]['/Subtype'] == '/Form':
                                    use_of_images = 'YES'
                                    image_counter += 1
                    IMAGES = image_counter

                    nr_of_characters = 0
                    rawText = parser.from_file(root + '/' + entry)

                    n1 = ' [0-9]+ years'  # 50 years
                    n2 = ' [1-2][0-9][0-9][0-9] '  # 2020
                    n3 = '[0-9]+ percent'  # 15 percent
                    n4 = '[0-9]+\%'  # 97%
                    n5 = '[0-9]+\.[0-9]+\%'  # 99.5%
                    n6 = '\$[0-9]+'  # $12
                    n7 = '[0-9]+\.[0-9]+ +[a-z]+'  # 1.1 billion
                    n8 = '[0-9]+\,[0-9]+ +[a-z]+'  # 5,000 people
                    n9 = '[0-9]+\.[0-9]+ +[a-z]+'  # 5.000 people
                    n10 = ' [0-9]+ +[a-z]+ '  # 15 respondents

                    clean_text = rawText['content'].replace('\n', '')
                    new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                                           clean_text)  # adds whitespace after between small and captial letter
                    good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ',
                                            new_sentences)  # adds whitespace after . and ,
                    normal_sentences = re.sub(r'\\xa0', r' ', good_sentences)  # changes \xa0 to whitespace
                    great_sentences = re.sub(r'[-]', r'', normal_sentences)
                    free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2',
                                            great_sentences)  # adds whitespace between number and letters
                    links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ",
                                   free_sentences)  # remove space between link
                    links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link

                    number_list = re.compile("(%s|%s|%s|%s|%s|%s|%s|%s|%s|%s)" % (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)).findall(links2)
                    email_addresses = re.compile("(%s)" % ('[a-z]+@[a-z]+\.com')).findall(str(links2))

                    NUMBERS = len(number_list)

                    try:
                        list_of_nr = rawText['metadata']['pdf:charsPerPage']
                        for nr_words_per_page in list_of_nr:
                            nr_of_characters += int(nr_words_per_page)
                        nr_of_characters = nr_of_characters

                        rawList = rawText['content'].splitlines()
                        while '' in rawList: rawList.remove('')
                    except:
                        continue

                    if directory[7] == firm_name[0]:
                        BRANDNAME = directory[7]
                        URL = firm_name[-1].strip('\n')
                        print(entry, BRANDNAME, URL, nr_of_pages, nr_of_characters, IMAGES, NUMBERS)
                        writer.writerow(
                            [BRANDNAME, URL, FILENAME, TITLE, AUTHORS, email_addresses, nr_of_pages, nr_of_characters,
                         use_of_images, IMAGES, NUMBERS])

            image_counter = 0

read_file = pd.read_csv(r'white_papers.csv')
read_file.to_excel(r'whitepapers.xlsx', index=None, header=True)

