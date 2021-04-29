import os
from tqdm import tqdm

with open("papers_divided_by_topic.csv", "r") as r:
    brand_and_pdfs = r.readlines()
    pdfs = []
    for line in brand_and_pdfs:
        first, second, brand, pdf = line.split(",")
        pdf = pdf.strip('\n')
        pdfs.append(pdf)

    for root, dirs, files in os.walk(r'C:\pdf_consultancy_firms', topdown=False):
        parent_folder_location = r'C:\pdf_consultancy_firms\CLASSIFIED_PAPERS'
        if not os.path.exists(parent_folder_location): os.mkdir(parent_folder_location)
        directory = root.split('\\')
        brandname = directory[-1]
        for file in tqdm(files):
            if file in pdfs:
                path = os.path.join(parent_folder_location, brandname)
                print(path)
                # if not os.path.exists(path): os.makedirs(path)
                # os.rename("pdfs_consultancy_firms/" + brandname + '\' + file,
                #           path + '\' + file)