import csv
from tika import parser

# import pdfReader as pdfReader
import pdftotext
import pandas as pd
from PyPDF2 import PdfFileReader
import os
import PyPDF2


# entries = os.listdir(r'white_papers')
# with open('white_papers.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["BRANDNAME", "URL", "TITLE", "AUTHORS", "NBR. OF PAGES", "NBR. OF WORDS"])
#
#     for entry in entries:
#         with open(r"white_papers/"+ entry, "rb") as f:
#             pdf = pdftotext.PDF(f)
#             with open('white_papers.csv', 'a', newline='') as file:
#                 nr_of_pages = len(pdf)
#                 writer.writerow(["", "", "", "", nr_of_pages, ""])
#
# read_file = pd.read_csv(r'white_papers.csv')
# read_file.to_excel(r'whitepapers.xlsx', index=None, header=True)

entries = os.listdir(r'white_papers')

with open('white_papers.csv', 'w', newline='') as file:
    firm_names = []
    firm_urls = []
    f = open("firms_and_url.txt", "r")
    firms = f.readlines()
    for i in firms:
        firm = i.split(",")
        firm_names.append(firm[0])
        firm_urls.append(firm[1])
    writer = csv.writer(file)
    writer.writerow(["BRANDNAME", "URL", "TITLE", "AUTHORS", "NBR. OF PAGES", "NBR. OF WORDS"])
    for entry in entries:
        print(entry)
        # with open(r"white_papers/"+ entry, "rb") as f:
        rawText = parser.from_file(r"white_papers/" + entry)
        rawList = rawText['content'].splitlines()
        for i in firm_names:
            for s in rawList:
                if i in s:
                    print(i)


# f = open("firms_and_url.txt", "r")
# firms = f.readlines()
# for firm in firms:
#     firm = firm.split(",")
#     print(firm[0])

            # pdf_information = PdfFileReader(f, strict=False)

            # information = pdf_information.getDocumentInfo()
            # print(entry, information)