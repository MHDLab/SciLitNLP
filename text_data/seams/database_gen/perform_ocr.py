#%%

import pandas as pd
import sqlite3
import os
from shutil import copyfile

# Import libraries
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

Image.LOAD_TRUNCATED_IMAGES = True
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def tesseract_ocr(PDF_file, temp_image_folder):

    # try:

    '''
    Part #1 : Converting PDF to images
    '''

    # Store all the pages of the PDF in a variable
    pages = convert_from_path(PDF_file, 500)


    # Iterate through all the pages stored above
    for image_counter, page in enumerate(pages):

        # Declaring filename for each page of PDF as JPG

        filename = "page_"+str(image_counter)+".jpg"

        # Save the image of the page in system
        page.save(os.path.join(temp_image_folder, filename), 'JPEG')


    '''
    Part #2 - Recognizing text from the images using OCR
    '''

    output_text = ''
    # Iterate from 1 to total number of pages
    for i in range(len(pages)):

        filename = os.path.join(temp_image_folder,"page_"+str(i)+".jpg")

        # Recognize the text as string in image using pytesserct
        im = Image.open(filename)
        text = str(pytesseract.image_to_string(im))
        text = text.replace('-\n', '')

        # Delete image to save space
        os.remove(filename)

        output_text += text

    return output_text





import git
repopath = git.Repo('.', search_parent_directories=True).working_tree_dir
data_folder = os.path.join(repopath, 'data')

con = sqlite3.connect(os.path.join(data_folder, 'seamsnlp_final.db'))
cursor = con.cursor()

df_meta = pd.read_sql_query("SELECT * FROM metadata", con, index_col='ID')

seams_pdf_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs\NoCovers'

temp_image_folder = os.path.join(data_folder, 'tmp')

if not os.path.exists(temp_image_folder):
    os.mkdir(temp_image_folder)

#%%
cursor.execute('select * from texts')

names = list(map(lambda x: x[0], cursor.description))

if 'OCR_text' not in names:
    print("creating ocr_text column")
    cursor.execute("ALTER TABLE texts ADD COLUMN OCR_text TEXT")

#%%
ids = range(361,409)
# ids = df.index

exists_list = []

for ID, row in df_meta.loc[ids].iterrows():
    relative_fp = row['Filepath']
    if relative_fp != None:
        # filepath = df['Filepath'][id]
        filepath = os.path.join(seams_pdf_folder, relative_fp)
        print('OCRing file: ' + filepath)

        filepath = os.path.join(seams_pdf_folder, filepath)

        # #This allows for long filenames on windows, not sure the effect on other os
        # #https://stackoverflow.com/questions/29557760/long-paths-in-python-on-windows
        # filepath = "\\\\?\\" + filepath.replace('/', '\\')

        exists = os.path.exists(filepath)
        exists_list.append(exists)

        if exists:
            text = tesseract_ocr(filepath, temp_image_folder)
            query = """UPDATE texts SET OCR_text= (?) WHERE ID = (?)"""
            cursor.execute(query, (text, ID))
            con.commit()
        else:
            print('could not find text: ')
            print(filepath)

df_exists = pd.Series(exists_list, index=ids)

#%%

#Display missing files
# df['Filepath'].where(df_exists == False).dropna().to_list()

# %%
