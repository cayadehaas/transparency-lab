import fitz
import os
from tqdm import tqdm
import pytesseract
import csv
import pandas as pd
image_text = []
from PIL import Image
with open('image_text.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FILENAME", 'IMAGE TEXT'])
    for root, dirs, files in os.walk('F:\/consultancy_firms_pdfs/WHITEPAPERS/accenture/', topdown=False):
        directory = root.split('\\')
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        for filename in tqdm(files):
            pdf_file = filename
            doc = fitz.open(root + pdf_file) # open pdf files using fitz bindings
            for i in range(len(doc)):
                for img in doc.getPageImageList(i):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:  # this is GRAY or RGB
                        pix.writePNG("p%s-%s.png" % (i, xref))
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.writePNG("p%s-%s.png" % (i, xref))
                        pix1 = None
                    pix = None
                    image_text.append(pytesseract.image_to_string(r'p%s-%s.png' % (i, xref)))

            writer.writerow([filename, image_text])
            image_text = []


read_file = pd.read_csv(r'image_text.csv')
read_file.to_excel(r'image_text.xlsx', index=None, header=True)