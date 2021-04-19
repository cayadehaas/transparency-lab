from collections import Counter
import pandas as pd
from tqdm import tqdm
import os
import re
from pdf2image import convert_from_path
import pytesseract
import spacy
nlp = spacy.load('/Users/cayadehaas/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-3.0.0')
POS_freq = Counter()

for root, dirs, files in os.walk(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/whitepapers_test', topdown=False):
    directory = root.split('/')
    for file in tqdm(files):
        tags = []
        pdf_file = root + '/' + file
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

            for token in nlp(li2):
                tags.append((token.tag_, token.pos_))
                POS_freq.update(tags)
        POS_freq.most_common(3)

# table = pd.DataFrame(columns=['Finegrained POS-tag', 'Universal POS-Tag', 'Occurrences', 'Relative Tag Frequency (%)', 'Most freq. tokens', 'Least freq. token'])
# s = sum(POS_freq.values())
# for thing in POS_freq.most_common(10):
#   tag = thing[0][0]
#   tags = [token.text
#           for tweet in files for token in tweet
#           if (token.tag_ == tag and not token.like_url)]
#   tags_freq = Counter(tags)
#   common_tokens = tags_freq.most_common(3)
#   common_tokens = ', '.join([token[0] for token in common_tokens])
#   uncommon_token = tags_freq.most_common()[-1][0]
#
#   pos = thing[0][1]
#   freq = thing[1]
#   rel_tag_freq = str(round((freq / s) * 100, 2)) + '%'
#   z = pd.DataFrame({'Finegrained POS-tag': [tag],
#                     'Universal POS-Tag': [pos],
#                     'Occurrences': [freq],
#                     'Relative Tag Frequency (%)': [rel_tag_freq],
#                     'Most freq. tokens': [common_tokens],
#                     'Least freq. token': [uncommon_token]})
#   table = pd.concat([table, z], ignore_index=True)

