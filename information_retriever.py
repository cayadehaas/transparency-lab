import csv
from tika import parser
import textract, re
import pdftotext
import pandas as pd
import os
from PyPDF2 import PdfFileReader
import pdftitle
import glob
# brandname, url, title , authors, nbr of pages, nbr of words, nbr of illustrations, nbr of graphs, nbr of numbers/percentages

with open('white_papers.csv', 'w', newline='') as file:
    image_counter = 0
    writer = csv.writer(file)
    writer.writerow(["BRANDNAME", "URL", "FILENAME", "TITLE", "AUTHORS", "NBR. OF PAGES", "NBR. OF CHARACTERS", "USE OF IMAGES", "NBR. OF IMAGES", "NBR. OF NUMBERS"])

    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms/WHITEPAPERS',
                                 topdown=False):
        directory = root.split('/')

        f = open("firms_and_url.txt", "r")
        firms = f.readlines()
        for firm in firms:
            for file in files:
                firm_name = firm.split(',')
                if '.pdf' in file:
                    with open(root + "/" + file, "rb") as f:
                        FILENAME = file
                        pdf = pdftotext.PDF(f)
                        nr_of_pages = len(pdf)

                    pdf_toread = PdfFileReader(open(root + "/" + file, "rb"))
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
                    rawText = parser.from_file(root + '/' + file)

                    clean_text = rawText['content'].replace('\n', '')
                    terms = clean_text.split(' ')
                    NUMBERS = sum(c.isdigit() for c in terms)

                    try:
                        list_of_nr = rawText['metadata']['pdf:charsPerPage']
                        for nr_words_per_page in list_of_nr:
                            nr_of_characters += int(nr_words_per_page)
                        nr_of_characters = nr_of_characters



                        rawList = rawText['content'].splitlines()
                        while '' in rawList: rawList.remove('')
                    except:
                        continue
                    try:
                        if directory[7] == firm_name[0]:
                            BRANDNAME = directory[7]
                            URL = firm_name[-1].strip('\n')
                            print(file, BRANDNAME, URL, nr_of_pages, nr_of_characters, IMAGES, NUMBERS)
                            writer.writerow([BRANDNAME, URL, FILENAME, TITLE, AUTHORS, nr_of_pages, nr_of_characters, use_of_images, IMAGES, NUMBERS])
                    except:
                        continue

            image_counter = 0

read_file = pd.read_csv(r'white_papers.csv')
read_file.to_excel(r'whitepapers.xlsx', index=None, header=True)

