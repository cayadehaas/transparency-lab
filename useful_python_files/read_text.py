import os
from tqdm import tqdm
from tika import parser
import re
from PyPDF2 import PdfFileReader
counter = 0
for root, dirs, files in os.walk('F:\pdf_consultancy_firms\CLASSIFIED_PAPERS/Digital transformation',
                                 topdown=False):
    for filename in tqdm(files):
        if ".pdf" in filename:
            f = os.path.join(root, filename)
            # pdf = PdfFileReader(f, "rb")
            rawText = parser.from_file(root + '/' + filename)
            clean_text = rawText['content'].replace('\n', '')
            rawList = rawText['content'].splitlines()
            while '' in rawList: rawList.remove('')
            while ' ' in rawList: rawList.remove(' ')
            while '\t' in rawList: rawList.remove('\t')
            text = ''.join(rawList)
            text = text.lower()
            print(text)
            # text = ''
    #         try:
    #             for i in range(pdf.getNumPages()):
    #                 page = pdf.getPage(i)
    #                 text = text + page.extractText()
    #                 text = text.replace('\n', ' ')
    #                 text = text.lower()
    #                 n_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    #                 g_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', n_sentences)  # adds whitespace after . and , 
    #                 y_sentences = re.sub(r'\\xa0', r' ', g_sentences)  # changes \xa0 to whitespace 
    #                 b_sentences = re.sub(r'% 20', r'', y_sentences)  # changes %20 to whitespace 
    #                 t_best_sentences = re.sub(r'% 40', r'', b_sentences)  # changes %40 to whitespace 
    #                 gr_sentences = re.sub(r'[-]', r'', t_best_sentences)
    #                 grd_sentences = re.sub(r'Š', r' ', gr_sentences)
    #                 grdf_sentences = re.sub(r'™', r"'", grd_sentences)
    #                 fr_sentences = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2',
    #                                       grdf_sentences)  # adds whitespace between number and letters 
    #                 li = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", fr_sentences)  # remove space between link 
    #                 li2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", li)
    #         except:
    #             counter += 1
    #             continue
    # print(counter)
