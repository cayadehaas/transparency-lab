import fitz
import os
from tqdm import tqdm
import pytesseract
from PIL import Image
import io


for root, dirs, files in os.walk('F:\/consultancy_firms_pdfs/WHITEPAPERS/accenture/', topdown=False):
    directory = root.split('\\')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # print(pytesseract.image_to_string(r'D:\examplepdf2image.png'))
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
                print(pytesseract.image_to_string(r'p%s-%s.png' % (i, xref)))