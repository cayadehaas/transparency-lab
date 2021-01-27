import os
import PyPDF2
import textract

terms = ['affidavit ', 'applicant ', 'confidential ', 'businessman ', 'disbursement ', 'retainer ', 'perks', 'bans ',
         'subservices', 'microinteractions ', 'interbank ', 'burdens ', 'makercentered ', 'recalibrate ', 'aggregators ',
         'interagency ', 'laggards ', 'borrowers ', 'profiles ', 'advisers ''currencies ', 'unsubsidized ', 'retrievals ',
         'provocative ', 'webcasts ', 'commensurate ', 'reinsurance ', 'errorseeding ', 'cloning ', 'regasification ',
         'hackathons', 'prohibitions', 'anonymized', 'radiation', 'fiscalquarter', 'apprenticeship ', 'reservesswift',
         'sublease', 'dictates', 'laborintensive', 'biannual', 'resiliency', 'rebates', 'federation', 'inservice',
         'dictates', 'referrals', 'optioneering', 'reimbursement', 'stipulation', 'payables', 'labelers', 'counsellor',
         'vectoring', 'sophistication', 'licensees', 'nonbanks', 'lawsuits', 'millennials', 'remediation', 'subsidy',
         'provocative', 'nonqualified', 'remuneration', 'prioritizations', 'nonappealable', 'payerspecific', 'brochure',
         'webinars', 'antidilution', 'exteriores', 'clinicians', 'interdepartmental', 'preemption', 'naval',
         'pertaining', 'spun ', 'putative', 'envisioning', 'codex', 'lawsuits', 'unsecured', 'rehabilitate',
         'nonaccelerated', 'promissory', 'unilateral', 'nodule', 'micropayment ', 'microtransaction', 'roboadvisers',
         'microsubscription', 'exemptive', 'upskilling']
# entries = os.listdir(r'white_papers_test')
parent_folder_location = r'pdfs_consultancy_firms/NON_WHITEPAPERS/'
if not os.path.exists(parent_folder_location): os.mkdir(parent_folder_location)

for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/more_whitepapers', topdown=False):
    directory = root.split('/')
    for entry in files:
        pdf = open('more_whitepapers/' + entry, 'rb')
        try:
            file_reader = PyPDF2.PdfFileReader(pdf)
            for i in range(file_reader.numPages):
                pageObj = file_reader.getPage(i)
                text = pageObj.extractText()
                for term in terms:
                    if term in text:
                        print(entry)
                        print(term)

            text = textract.process(
                'pdfs_consultancy_firms/WHITEPAPERS/consultarc/Whitepaper_Being_Digital_means_Being_more_Human.pdf',
                method='pdfminer')
            readable_text = str(text)
            for term in terms:
                if term in readable_text:
                    print(entry)
                    print(term)
        except:
            text = textract.process(
                'more_whitepapers/'+ entry,
                method='pdfminer')
            readable_text = str(text)
            for term in terms:
                if term in readable_text:
                    print(entry)
                    print(term)

