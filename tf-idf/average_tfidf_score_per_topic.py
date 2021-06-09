import pytesseract as pytesseract
from PyPDF2 import PdfFileReader
from matplotlib.pyplot import stem
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re
import csv
from pdf2image import convert_from_path
from tqdm import tqdm
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import string
from tika import parser

def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    english_stopwords = stopwords.words('english')
    rawText = parser.from_file(pdf)
    clean_text = rawText['content'].replace('\n', '')
    rawList = rawText['content'].splitlines()
    while '' in rawList: rawList.remove('')
    while ' ' in rawList: rawList.remove(' ')
    while '\t' in rawList: rawList.remove('\t')
    text = ''.join(rawList)
    text = text.lower()
    new_sentences = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                           text)  # adds whitespace after between small and captial letter
    good_sentences = re.sub(r'(?<=[.,?!%:])(?=[^\s])', r' ', new_sentences)  # adds whitespace after . and ,
    normal_sentences = re.sub(r'\\xa0', r' ', good_sentences)  # changes \xa0 to whitespace
    best_sentences = re.sub(r'% 20', r'', normal_sentences)  # changes %20 to whitespace
    the_best_sentences = re.sub(r'% 40', r'', best_sentences)  # changes %40 to whitespace
    great_sentences = re.sub(r'[-]', r'', the_best_sentences)
    free_sentences = re.sub(r'([0-9])([a-z|A-Z])', r'\1 \2',
                            great_sentences)  # adds whitespace between number and letters
    links = re.sub(r"((www\.) ([a-z]+\.) (com))", r" \2\3\4 ", free_sentences)  # remove space between link
    links2 = re.sub(r"(([A-Za-z]+@[a-z]+\.) (com))", r" \2\3 ", links)  # remove space between link
    links3 = re.sub(r"(www)[A-Za-z]+(com)", r" ", links2)  # remove links
    links4 = re.sub(r"(http)[A-Za-z]+", r" ", links3)  # remove links
    links5 = re.sub(r"[A-Za-z]+(http)[A-Za-z]+", r" ", links4)  # remove links
    links6 = re.sub(r"[A-Za-z]+(http)", r" ", links5)  # remove links
    links7 = re.sub(r"[A-Za-z]+(https)", r" ", links6)  # remove links
    links8 = re.sub(r"[A-za-z]+(com)", r" ", links7)

    tokenized_text = nltk.word_tokenize(links8)

    nltk.pos_tag(tokenized_text)
    words = []
    for word, tag in nltk.pos_tag(tokenized_text):
        # NN noun, singular,  NNS noun plural, NNP proper noun singular , NNPS proper noun plural
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            words.append(word)
    table = {ord(char): '' for char in string.punctuation}
    cleaned_messy_sentence = []
    for messy_word in words:
        cleaned_word = messy_word.translate(table)
        cleaned_messy_sentence.append(cleaned_word)

    without_stopwords = []

    for token in cleaned_messy_sentence:
        if token.isalpha() and len(token) > 3 and len(token) < 18 and token not in english_stopwords and token not in [
            'infosys', 'aarete', 'consultarc', 'abeam', 'accenture', 'acquisconsulting', 'aeqglobal', 'alacrita',
            'alixpartner', 'altiusdata', 'altvil', 'altumconsulting', 'alvarezandmarsal', 'amaris', 'analysisgroup',
            'ananyaaconsultingltd', 'ankura', 'appliedvaluegroup', 'appsbroker', 'astadia', 'auriemma',
            'aurigaconsulting', 'avanade', 'avanceservices', 'alvarez', 'wwwtwittercom', 'wwwmercercom', 'htmlhttps',
            'wwwgartnercom', 'wwwcognizantcom', 'wwwfticonsultingcom', 'ofwat', 'wwwirishtimescom', 'wwwnytimescom',
            'wwwsmithsonianmagcom', 'wwwslatecom', 'wwwhistorycom', 'nypost', 'wwwdanyacom', 'wwworaclecom',
            'wwwcelentcom', 'wwwreuterscom', 'wwwpressreadercom', 'wwwemarketercom', 'orgsdgshttp', 'articlehttp',
            'wwwigiglobalcom', 'wwwprmaconsultingcom', 'prmaconsultingcom', 'dotcom', 'wwwchartiscom', 'chartiscom',
            'wwwvmwarecom', 'wwwadnewscom', 'wwwptccom', 'wwwuipathcom', 'wwwbarronscom', 'wwwbhpcom',
            'wwweconomistcom', 'cfds', 'wwwtechradarcom', 'wwwpegacom', 'wwwjohnhancockcom', 'jawwy', 'wwwchinacom',
            'bahcom', 'fminetcom', 'hcit', 'hipaa', 'olse', 'gartnercom', 'wwwlittlercom', 'btscom', 'cognizantcom',
            'bain', 'bainbridgeconsulting', 'wwwoddscheckercom', 'wwwmagistocom', 'bakertilly', 'wwwmeedcom',
            'marshcom', 'paconsultingcom', 'bateswhite', 'bdo', 'blayzegroup', 'bluematterconsulting', 'boozallen',
            'bcg', 'bsienergyventures', 'boxtwenty', 'boxfusionconsulting', 'bpi', 'bridgespan', 'bts', 'bwbconsulting',
            'capgemini', 'capitalone', 'cssfundraising', 'cellohealthbioconsulting', 'censeoconsulting',
            'centricconsulting', 'crai', 'wwwpwccom', 'pwccom', 'infosyscom', 'wwwkpmginstitutescom', 'wwwkpmgcom',
            'wwwpreview', 'abeamcom', 'wwwabeamcom', 'wwwcornerstonecom', 'wwwkovaioncom', 'wwwwsjcom',
            'wwwtheatlanticcom', 'wwwnydailynewscom', 'wwwsmhcom', 'wwwextremetechcom', 'wwwprnewswirecom',
            'wwwwealthmanagementinsightscom', 'wwwnovantascom', 'novantascom', 'wwwmckinseycom', 'eycom', 'wwwefmacom',
            'wwwcraicom', 'craicom', 'wwwanandtechcom', 'wwwphonearenacom', 'lutwuibfhttp', 'urlhttp', 'wwwstrtradecom',
            'wwwmiebachcom', 'miebachcom', 'wwwhfsresearchcom', 'wwwrisiinfocom', 'wwwgooglecom', 'wwwendeavoreagcom',
            'wwwshellcom', 'wwwglasslewiscom', 'wwwbpcom', 'wwwbilgikentcom', 'wwwgminsightscom', 'wwwvircom', 'cfpb',
            'wwwchinanewscom', 'ichom', 'vbhc', 'orgindex', 'jbgsmithcom', 'wwwappfuturacom', 'fmcg', 'ofcom',
            'wwwriskukcom', 'wwwavanadecom', 'orgwww',
            'clancy', 'clearedgeconsulting', 'teneocabinetdncom', 'wwwmedifuturecom', 'standagencycom',
            'wwwnanowerkcom', 'wwwgeccfcom', 'petitionreviewcom', 'clearviewconsulting', 'cleolifesciences',
            'cobaltrecruitement', 'comsecglobal', 'concentra', 'consultingpoint', 'copehealthsolutions', 'cornerstone',
            'corporate value', 'crowe', 'css-cg', 'curtins', 'dalberg', 'deallus', 'dean', 'dhg', 'dmt-group',
            'doctors-management', 'eamesconsulting', 'eastwoordandpartners', 'subjectmailto', 'wwwoliverwymancom',
            'oliverwymancom', 'wwwinfosysbpmcom', 'wwwinfosyscom', 'wwwbloombergcom', 'wwwdeloittecom',
            'wwwgelbconsultingcom', 'intothe',
            'orgnanoclastsemiconductorsoptoelectronicsgraphenegivesyouinfraredvisioninacontactlenshttp',
            'timesfasterusingstandardcmosprocesseshttp', 'wwwcompoundchemcom', 'wwwspaniagtacom', 'wwwzdnetcom',
            'wwwciocom', 'healthcarehttp', 'wwwalizilacom', 'govhttp', 'wwwpaysafecom', 'wwwjdpowercom',
            'wwwvoanewscom', 'wwwthevergecom', 'wwwwiredcom', 'wwwdatarobotcom', 'cshtml', 'wwwdmtgroupcom', 'ebit',
            'wwwscjohnsoncom', 'wwwgallupcom', 'wwwficocom', 'nber', 'wellesley', 'wwwpremierinccom', 'wwwgsmarenacom',
            'schenck', 'hartmut', 'mahesh', 'bdocom', 'wwwteneocom', 'teneocom', 'conferencewww', 'wwwcityamcom',
            'baincom',
            'eca-uk', 'efficioconsulting', 'dalbergcom', 'putercom', 'pointbcom', 'egonzehnder', 'elementaconsulting',
            'elixirr', 'endeavormgmt', 'enterey', 'establishinc', 'eunomia', 'everstgrp', 'everis', 'ey', 'fichtner',
            'flattconsulting', 'fletcherspaght', 'flint-international', 'fminet', 'fsg', 'fticonsulting', 'ajg',
            'gallup', 'gartner', 'gazellegc', 'gehealthcare', 'gep', 'glg', 'grantthornton', 'wwwlinkedincom',
            'morehttps', 'wwwprotiviticom', 'protiviticom', 'userinfosyswww', 'wwwbassberrysecuritieslawexchangecom',
            'aecom', 'golinharriscom', 'wwwendeavormgmtcom', 'endeavormgmtcom', 'wwwforbescom', 'alixpartners',
            'llpemail', 'ieee', 'wwwdanyacom', 'wwwfinextracom', 'wwwajgcom', 'ajgcom', 'wwwmulesoftcom',
            'wwwcapgeminicom', 'wwwcoindeskcom', 'wwwncbacom', 'wwwcelentcom', 'wwwhttps', 'wwwfortnightlycom',
            'wwwidccom', 'demohttp', 'wwwtheguardiancom', 'fticonsultingcom', 'wwwamazoncom', 'wwwsljcom', 'wwwgsmacom',
            'wwwutilitydivecom', 'iinncc', 'wwwhrmreportcom', 'wwwrepsolcom', 'wwwbrinknewscom', 'wwwmacrumorscom',
            'wwwbdocom', 'wwwmarkitcom', 'wwwjabiancom', 'wwwthedrumcom', 'jabiancom', 'bcgplatinioncom',
            'pluralstrategycom', 'wwwcnncom', 'classiccuf', 'hong', 'wwwakamaicom', 'pearlmeyercom', 'hrgartner',
            'ndycom', 'wwwalixpartnerscom', 'alixpartnerscom', 'lekcom', 'whitelanecom', 'mmccom', 'camberviewcom',
            'grm-consulting', 'healthadvances', 'ankuracom', 'healthdimensionsgroup', 'wwwaltvilcom', 'fwww',
            'altvilcom', 'hiltonconsulting', 'horvath-partners', 'huronconsultinggroup', 'i-pharmconsulting', 'ibm',
            'ibuconsulting', 'infiniteglobal', 'infinityworks', 'infuse', 'innercircleconsulting', 'innogyconsulting',
            'intellinet', 'iqvia', 'jabian', 'jdxconsulting', 'atkearney', 'klifesciences', 'kirtanaconsulting',
            'knowledgecapitalgroup', 'kovaion', 'wwwbusinessinsidercom', 'wwwbcgcom', 'bcgcom', 'wwwaccenturecom',
            'accenturecom', 'wwwfacebookcom', 'wwwpearlmeyercom', 'wwwnaturecom', 'wwwspaniagtacom', 'wwwlekcom',
            'lekcom', 'wwweverestgrpcom', 'wwwfnlondoncom', 'wwwondeckcom', 'wwwcbinsightscom', 'wwwtencentcom',
            'wwwlsegcom', 'wwwoxeracom', 'wwwreuterscom', 'wwwstatistacom', 'wwwmycustomercom', 'wwwnovartiscom',
            'wwwpfizercom', 'wwwnewsdaycom', 'wwwcebglobalcom', 'wwwegonzehndercom', 'wwwsljcom', 'gelbconsultingcom',
            'gagarrttnneerr', 'wwwgoodreadscom', 'wwwndycom', 'comhttp', 'kuntze', 'wwwapplecom', 'wwwqacom',
            'elqcampaignidhttp', 'fsieg', 'shamsabadi', 'wwwboozallencom', 'idns', 'wwwsynpulsecom', 'btscom', 'gcom',
            'gtnr', 'navientcom', 'healthadvancescom', 'dwtcom', 'wwwheartflowcom', 'bcgfedcom', 'appsbrokercom',
            'pearlymeyercom',
            'kpmg', 'lda-design', 'wwwdoterracom', 'pragmaukcom', 'medtechinnocom', 'gmailcom', 'lek', 'lexamed',
            'lscconnect', 'magenic', 'marakon', 'marsandco', 'mmc', 'mcchrystalgroup', 'mckinsey', 'medcgroup',
            'mercer', 'methodllp', 'bradbrookconsulting', 'mmgmc', 'mlmgroup', 'mottmac', 'navigant', 'neueda',
            'newtoneurope', 'nmg-group', 'ndy', 'novantas', 'ocs-consulting', 'occstrategy', 'olivehorse',
            'oliverwyman', 'opusthree', 'oxera', 'oxfordcorp', 'paconsulting', 'cellohealthcom', 'pmcretail',
            'pearlmeyer', 'puralstrategy', 'pointb', 'wwwanalysysmasoncom', 'pearl', 'meyer', 'phphttp', 'index',
            'wwwhttps', 'wwwefmacom', 'wwwreuterscom', 'wwwcomscorecom', 'wwwnielsencom', 'htmlwww', 'wwwcnetcom',
            'wwwretailweekcom', 'wwwexperiancom', 'wwwgreenbizcom', 'wwwsmartbriefcom', 'wwwraybancom', 'htmlhttp',
            'wwwrochecom', 'wwwpcworldcom', 'wwwwalkerinfocom', 'wwwspglobalcom', 'wwwplexcom', 'wwwcsoonlinecom',
            'oetker', 'theo', 'rrighighttss', 'wwwlazardcom', 'ofse', 'ervs', 'bennettmob', 'wwwzteusacom',
            'wwwblackrockcom', 'wwwbaincom', 'acdpgnvfiimm', 'cios', 'ennewsroom', 'wwwceoactioncom', 'everestgrpcom',
            'emea', 'capgeminicom', 'deloittemxcom', 'iifcom',
            'portasconsulting', 'pragmauk', 'prmaconsulting', 'prophet', 'protiviti', 'providge', 'ptsconsulting',
            'publicconsultinggroup', 'putassoc', 'pwc', 'qa', 'red-badger', 'rhrinternationalconsultants', 'rise',
            'robertwest', 'rsmus', 'rsp', 'russelreynolds', 'sia-partners', 'slalom', 'ssandco', 'argentwm',
            'starlizard', 'sterlingassociates', 'sternstewart', 'londonstrategicconsulting', 'stroudinternational',
            'syneoshealth', 'wwwpaconsultingcom', 'wwwbakertillycom', 'wwwsiapartnerscom', 'siapartnerscom', 'wwweycom',
            'bakertillycom', 'wwwftcom', 'wwweverestgrpcom', 'wwwgtreviewcom', 'wwwreuterscom', 'wwwcnbccom',
            'wwwaccorhotelscom', 'htmhttp', 'helcom', 'wwwretaildivecom', 'wwwcmocom', 'wwwsalesforcecom',
            'wwwfastcompanycom', 'wwwnxtbookcom', 'wwwstatistacom', 'wwwbtscom', 'morehttp', 'langenhttp',
            'icclimitedwww', 'wwwfminetcom', 'wwwthelancetcom', 'wwwplanestatscom', 'wwwforrestercom', 'wwwssaandcocom',
            'ssaandcocom', 'subjecthttp', 'wwwnbcnewscom', 'wwwsascom', 'lehrke', 'gbso', 'wwwinfosysbpocom',
            'wwwstokcom', 'khttp', 'ftservices', 'orxadu', 'wwwadvisorycom', 'wwwpharmexeccom', 'wwwnstocom',
            'egonzehndercom', 'wwwcomsecglobalcom', 'comsecglobalcom',
            'synpulse', 'syntegragroup', 'wwwocbccom', 'mckinseycom', 'sdggroupcom', 'wwwrbscom', 'pegacom',
            'weworldwidecom', 'auhttp', 'talan', 'team-consulting', 'teamwill-consulting', 'tefen', 'idoc', 'chartis',
            'eckrothplanning', 'thoughtprovokingconsulting', 'transformconsultinggroup', 'trinity-partners',
            'ipeglobal', 'turnkeyconsulting', 'unboxed', 'vantagepartners', 'ivaldigroup', 'izientinc', 'miebach',
            'whitepaper', 'white', 'paper', 'deloitte', 'cognizant', 'capgemini', 'gartner', 'wyman', 'synpulsecom',
            'wwwomnicom', 'omnicom', 'greeleycom', 'httpgartner', 'compcom', 'oliver', 'https', 'http', 'pdfhttps',
            'pdfhttp', 'pdf', 'kpmgcom', 'wwwyoutubecom', 'mailto', 'deloittecom', 'wwwfleetownercom', 'combytellccom',
            'wwwceinetworkcom', 'wwwautoblogcom', 'wwwscmagazinecom', 'wwwbbccom', 'wwwinstagramcom', 'amcom',
            'wwwmicrosoftcom', 'wwwfireeyecom', 'wwwcpomagazinecom', 'wwwengadgetcom', 'erikberg', 'facebook',
            'twitter', 'instagram', 'refalgobottomrec', 'thing', 'html', 'richards', 'morethe', 'wwwquoracom',
            'rreesseerrvveedd', 'wwwxilinxcom', 'orghttp', 'wwwmarakoncom', 'decc', 'pati', 'teneointel',
            'wwwpovaddocom', 'wwwvaluewalkcom', 'ngam', 'wwwbtplccom', 'documenthelp', 'geisinger', 'pfml',
            'allsalariesyes', 'mercercom']:
            without_stopwords.append(token)
    text_without_stopwords = ' '.join(without_stopwords)
    return text_without_stopwords


# text = ''
    # for i in range(pdf.getNumPages()):
    #     page = pdf.getPage(i)
    #     text = text + page.extractText()
    # return text

def stem_tokenize(document):
    '''return stemmed words longer than 2 chars and all alpha'''
    tokens = [stem(w) for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens

def tokenize(document):
    '''return words longer than 2 chars and all alpha'''
    tokens = [w for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens

def build_corpus_from_dir(dir_path):

    corpus = []
    brandnames_and_filenames = []
    for root, dirs, filenames in os.walk(dir_path, topdown=False):
        directory = root.split('/')
        for name in filenames:
            f = os.path.join(root, name)
            brandname = directory[-1]
            filename = name
            brandnames_and_filenames.append([brandname, filename])

            if '.pdf' in name:
                # pdf = PdfFileReader(f, "rb")
                pdf = root + '/' + name
                document = pdf2text(pdf)
                corpus.append(document)

    return corpus, brandnames_and_filenames


if __name__ == '__main__':
    with open('../Travel & Transportation.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["BRANDNAME", "FILENAME", "AVERAGE TF-IDF SCORE"])

        corpus, brandnames_and_filenames = build_corpus_from_dir(r'/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdf_consultancy_firms/Travel & Transportation')
        vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop)
        vectorizer.fit(corpus)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()

        for index_brand, brand_file in enumerate(brandnames_and_filenames):
            for index, doc in enumerate(corpus):
                if index_brand == index:
                    brandname = brand_file[0]
                    filename = brand_file[1]
                    keywords = []
                    tf_idf_scores = []
                    scores = []
                    doc = ' '.join(doc.split())

                    tdm = vectorizer.transform([doc])
                    dense = tdm.todense()
                    episode = dense[0].tolist()[0]
                    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
                    for phrase, score in [(features[word_id], score) for (word_id, score) in sorted_phrase_scores][:50]:
                        keywords.append(phrase)
                        tf_idf_scores.append('{0} {1}'.format(phrase, score))
                        scores.append(score)
                        print('{0: <20} {1}'.format(phrase, score))
                    print()
                    try:
                        average_tf_idf_score = sum(scores) / len(scores)
                    except ZeroDivisionError:
                        print([brandname, filename])
                        # writer.writerow([brandname, filename])

                    print([brandname, filename, average_tf_idf_score])
                    writer.writerow([brandname, filename, average_tf_idf_score])

    read_file = pd.read_csv(r'../Travel & Transportation.csv')
    read_file.to_excel(r'Travel & Transportation.xlsx', index=None, header=True)


