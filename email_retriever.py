import os
from tqdm import tqdm
from tika import parser
import re

n1 = '[a-z]+@[a-z]+\.com'              #email


for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/whitepapers_test', topdown=False):
    directory = root.split('/')
    for entry in tqdm(files):
        pdf = open('whitepapers_test/' + entry, 'rb')
        rawText = parser.from_file('whitepapers_test/' + entry)
        rawList = rawText['content'].splitlines()
        while '' in rawList: rawList.remove('')
        while ' ' in rawList: rawList.remove(' ')
        while '\t' in rawList: rawList.remove('\t')
        text = ''.join(rawList)
        new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                               text)  # adds whitespace after between small and captial letter
        good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', new_sentences)  # adds whitespace after . and ,
        normal_sentences = re.sub(r'\\xa0', r' ', good_sentences)  # changes \xa0 to whitespace
        great_sentences = re.sub(r'[-]', r'', normal_sentences)
        free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2',
                                great_sentences)  # adds whitespace between number and letters
        links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", free_sentences)  # remove space between link
        links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link
        email_addresses = re.compile("(%s)" % (n1)).findall(str(links2))
