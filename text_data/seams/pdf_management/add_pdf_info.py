#%%
import PyPDF2
import pandas as pd
import os

from reportlab.pdfgen import canvas, textobject
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.rl_config import defaultPageSize


def gen_cover_page(fp_cover_temp, paper_meta):
    """Generate temporary coverpage with metadata"""

    
    

    styleSheet = getSampleStyleSheet()
    styleN = styleSheet['Normal']
    styleH = styleSheet['Heading1']
    styleH.alignment = TA_CENTER
    story = []
    styleN.alignment = TA_CENTER
    styleN.spaceAfter = 6

    c = canvas.Canvas(fp_cover_temp)
    header = paper_meta['Title']
    story.append(Paragraph(header, styleH))

    infolist = ['Author(s)', 'Session Name']
    for idx in infolist:
        s = idx + ': ' + str(paper_meta[idx])
        story.append(Paragraph(s, styleN))



    story.append(Paragraph('SEAM: ' + str(paper_meta['SEAM']) + ' (' + str(paper_meta['Year']) +')', styleN))

    story.append(Paragraph('SEAM EDX URL: ' + paper_meta['SEAM EDX URL'], styleN))


    # doi_link_string = 'SEAM DOI: <link href="https://www.osti.gov/scitech/search/filter-results:FD/semantic:' + paper_meta['DOI'] + '">' + str(paper_meta['DOI'])+ '</link>'
    # story.append(Paragraph( doi_link_string, styleN))

    story.append(Paragraph('EDX Paper ID' + ': ' + str(paper_meta.name), styleN))
 
    f = Frame(inch, inch, 6*inch, 9*inch, showBoundary=1)
    f.addFromList(story, c)
    c.save()


from reportlab.lib.colors import blue, black

def gen_header_page(fp_header_temp, paper_meta, media_box_page1):
    """
    Generates temporary header pdf to be merged with each page

    This is using the "nonplatypus" method right now which is clunky, and can't
    seemingly do underlined text easily for string. Not sure how easy it is to
    get platypus to make a signle string on a page at any location
    """

    page_height = float(media_box_page1[3])
    page_width = float(media_box_page1[2])
    dy = 20
    c = canvas.Canvas(fp_header_temp)

    font_size = 8
    c.setFontSize(font_size)

    linkstr = 'SEAM #'
    linkstr += str(paper_meta['SEAM'])
    linkstr += ' (' + str(paper_meta['Year']) + ')'
    
    otherstr = ', Session: ' + paper_meta['Session Name']

    link_width = c.stringWidth(linkstr)
    full_header_width = c.stringWidth(linkstr + otherstr)

    c.setFillColor(blue)

    c.drawString(page_width/2 - full_header_width/2, page_height - dy, linkstr)

    #https://www.hoboes.com/Mimsy/hacks/adding-links-to-pdf/
    hostlink = paper_meta['SEAM EDX URL']

    #height apparently 1.2 times font size
    #https://reportlab-users.reportlab.narkive.com/hYWca3nW/finding-out-the-width-height-of-text
    header_height = font_size*1.2 

    #bottom left and top right corners of rectangle
    linkRect = (
        page_width/2 - link_width/2,
        (page_height - dy),
        page_width/2 + link_width/2,
        (page_height - dy) + header_height
           )
    c.linkURL(hostlink, linkRect)    

    # Add other (nonlink) header text 

    c.setFillColor(black)

    c.drawString(page_width/2 - full_header_width/2 + link_width, page_height - dy, otherstr)

    # c.drawCentredString(page_width/2, page_height - dy, headerstr)
    
    

    c.save()

#%%

import warnings

#https://github.com/tdamdouni/Pythonista/blob/master/pdf/add-header-to-pdf.py

# https://forum.omz-software.com/topic/3966/adding-a-page-header-to-pdf-output/5

def add_cover_header(fp_in, fp_out, paper_meta, temp_folder):
    fp_header_temp = os.path.join(temp_folder, 'temp_header.pdf')
    fp_cover_temp = os.path.join(temp_folder, 'temp_cover.pdf')

    gen_cover_page(fp_cover_temp, paper_meta)

    with open(fp_in, 'rb') as pdf_file:
        pdf_file_reader = PyPDF2.PdfFileReader(pdf_file)
        media_box_page1 = pdf_file_reader.getPage(0).mediaBox
        gen_header_page(fp_header_temp, paper_meta, media_box_page1)
        with open(fp_header_temp, 'rb') as header_file, open(fp_cover_temp, 'rb') as cover_file:
            bg_page = PyPDF2.PdfFileReader(header_file).getPage(0)
            pdf_out = PyPDF2.PdfFileWriter()
            cover_file_reader = PyPDF2.PdfFileReader(cover_file)
            cover_page = cover_file_reader.getPage(0)
            pdf_out.addPage(cover_page)
            for i, page in enumerate(pdf_file_reader.pages):
                if page.extractText():  # Do not copy pages that have no text'
                    # assert page.mediaBox == media_box_page1
                    if page.mediaBox != media_box_page1:
                        warnings.warn("Warning: Page " + str(i) + " of file did not match size of first page")
                        
                    page.mergePage(bg_page)
                    pdf_out.addPage(page)
            if pdf_out.getNumPages():
                with open(fp_out, 'wb') as out_file:
                # Caution: All three files MUST be open when write() is called
                    pdf_out.write(out_file)
    # %%


if __name__ == '__main__':

    temp_folder = r'C:\Users\aspit\Git\MHDLab-Projects\NLP_MHD\pdf_management\test'

    fp_out = os.path.join(temp_folder, 'EDX pdf demo.pdf')

    seams_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs'
    df_meta = pd.read_csv(os.path.join(seams_folder, 'SEAMS_metadata.csv'), index_col=['ID'])
    df_meta['EDX Paper ID'] = df_meta.index

    df_meta['EDX url'] = r'https://edx.netl.doe.gov/dataset/seam-' + df_meta['SEAM'].apply(str)

    paper_meta = df_meta.iloc[1000]

    fp_in = os.path.join(seams_folder, paper_meta['Filepath'])

    

    add_cover_header(fp_in, fp_out, paper_meta, temp_folder)
