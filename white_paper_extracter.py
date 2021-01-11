import os
import PyPDF2
import textract


def title_searcher():
    """This function looks at the title of the pdf to find white papers and places them in another directory
        It looks if non white paper terms appear in the files and places those in another directory"""
    white_paper_folder_location = r'pdfs_consultancy_firms/WHITEPAPERS/'
    if not os.path.exists(white_paper_folder_location): os.mkdir(white_paper_folder_location)

    non_white_paper_folder_location = r'pdfs_consultancy_firms/NON_WHITEPAPERS/'
    if not os.path.exists(non_white_paper_folder_location): os.mkdir(non_white_paper_folder_location)

    white_paper = ['whitepaper', 'White Paper', 'Whitepaper', 'White paper', 'white-paper', 'white_paper', 'White_paper', 'White_Paper']
    terms = ['affidavit', 'applicant', 'confidential', 'businessman', 'disbursement', 'retainer']

    for root, dirs, files in os.walk('/Users/cayadehaas/PycharmProjects/Transparency-Lab/pdfs_consultancy_firms',
                                     topdown=False):
        directory = root.split('/')

        for filename in files:
            # Check if white paper in title
            if any(x in filename for x in white_paper):
                firm = directory[-1]
                path = os.path.join(white_paper_folder_location, firm)
                if not os.path.exists(path): os.makedirs(path)
                os.rename("pdfs_consultancy_firms/" + firm + '/' + filename,
                          path + '/' + filename)
            else:
                try:
                    pdf = open(root + '/' + filename, 'rb')
                    file_reader = PyPDF2.PdfFileReader(pdf)

                    for i in range(file_reader.numPages):
                        pageObj = file_reader.getPage(i)
                        text = pageObj.extractText()

                        if any(x in filename for x in white_paper):
                            firm = directory[-1]
                            path = os.path.join(white_paper_folder_location, firm)
                            if not os.path.exists(path): os.makedirs(path)
                            os.rename("pdfs_consultancy_firms/" + firm + '/' + filename,
                                      path + '/' + filename)

                        for term in terms:
                            #check if file contains non-white paper terms and add to directory
                            if term in text:
                                firm = directory[-1]
                                path = os.path.join(non_white_paper_folder_location, firm)
                                if not os.path.exists(path): os.makedirs(path)
                                os.rename("pdfs_consultancy_firms/" + firm + '/' + filename,
                                          path + '/' + filename)

                    text = textract.process('pdfs_consultancy_firms/' + directory[-1] + '/' + filename, method='pdfminer')
                    readable_text = str(text)
                    if any(x in filename for x in white_paper):
                        firm = directory[-1]
                        path = os.path.join(white_paper_folder_location, firm)
                        if not os.path.exists(path): os.makedirs(path)
                        os.rename("pdfs_consultancy_firms/" + firm + '/' + filename,
                                  path + '/' + filename)
                    for term in terms:
                        # check if file contains non-white paper terms and add to directory
                        if term in readable_text:
                            firm = directory[-1]
                            path = os.path.join(non_white_paper_folder_location, firm)
                            if not os.path.exists(path): os.makedirs(path)
                            os.rename("pdfs_consultancy_firms/" + firm + '/' + filename,
                                      path + '/' + filename)
                except:
                    continue


def main():
    title_searcher()


if __name__ == '__main__':
    main()
