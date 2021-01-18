import csv
from tika import parser
import textract, re
import pdftotext
import pandas as pd
import os
import glob
# brandname, url, title , authors, nbr of pages, nbr of words, nbr of illustrations, nbr of graphs, nbr of numbers/precentages

with open('white_papers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["BRANDNAME", "URL", "TITLE", "AUTHORS", "NBR. OF PAGES", "NBR. OF CHARACTERS"])

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
                        pdf = pdftotext.PDF(f)
                        nr_of_pages = len(pdf)
                    nr_of_characters = 0
                    rawText = parser.from_file(root + '/' + file)
                    try:
                        list_of_nr = rawText['metadata']['pdf:charsPerPage']
                        for nr_words_per_page in list_of_nr:
                            nr_of_characters += int(nr_words_per_page)
                        nr_of_characters = nr_of_characters
                        rawList = rawText['content'].splitlines()
                    except:
                        continue
                    try:
                        if directory[7] == firm_name[0]:
                            BRANDNAME = directory[7]
                            URL = firm_name[-1].strip('\n')
                            print(file, BRANDNAME, URL, nr_of_pages, nr_of_characters)
                            writer.writerow([BRANDNAME, URL, "TITLE", "AUTHORS", nr_of_pages, nr_of_characters])
                    except:
                        continue

read_file = pd.read_csv(r'white_papers.csv')
read_file.to_excel(r'whitepapers.xlsx', index=None, header=True)

