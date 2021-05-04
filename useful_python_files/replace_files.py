import os
from tqdm import tqdm

with open("digital_transformation_papers.csv", "r") as r:
    brand_and_pdfs = r.readlines()
    pdfs = []
    for line in brand_and_pdfs:
        brand, pdf = line.split(",")
        pdf = pdf.strip('\n')
        pdfs.append(pdf)

    for root, dirs, files in os.walk(r'F:\pdf_consultancy_firms\CLASSIFIED_PAPERS', topdown=False):
        parent_folder_location = r'F:\pdf_consultancy_firms\CLASSIFIED_PAPERS\Digital transformation'
        if not os.path.exists(parent_folder_location): os.mkdir(parent_folder_location)
        directory = root.split('\\')
        brandname = directory[-1]
        for file in tqdm(files):
            if file in pdfs:
                path = os.path.join(parent_folder_location, brandname)
                if not os.path.exists(path): os.makedirs(path)
                # try: #UNDO REPLACING FILES
                #     os.rename(path + '/' + file, r"F:\pdf_consultancy_firms/" + brandname + '/' + file)
                # except FileNotFoundError:
                #     continue
                try:
                    os.rename(r"F:\pdf_consultancy_firms/CLASSIFIED_PAPERS/" + brandname + '/' + file,
                          path + '/' + file)
                except FileNotFoundError:
                    print(file)
                    continue
