#%%

import pandas as pd
import shutil
import os

from add_pdf_info import add_cover_header


#Use the old seams pdf folder to get consistent filename lengths. 
old_seams_pdf_folder =  r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs\Updated SEAMs'

seams_pdf_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs Archive\PDF Cleanup 2'

output_folder = r'C:\Users\aspit\Desktop\test seams'

df_meta = pd.read_csv(os.path.join(seams_pdf_folder, 'SEAMS_metadata.csv'), index_col=['ID'])


#%%

df_meta = df_meta.where(df_meta['SEAM']==11).dropna(subset=['Filepath'])#.iloc[0:5]
df_meta['SEAM'] = df_meta['SEAM'].astype(int) #WTF

df_meta
#%%



yeardict = {2: 1961, 3:	1962, 4:	1963,5:	1964,6:	1965,7:	1966,8:	1967,9:	1968,10:	1969,11:	1970,12:	1972,13:	1973,14:	1974,15:	1976,16:	1977,17:	1978,18:	1979,19:	1981,20:	1982,21:	1983,22:	1984,23:	1985,24:	1986,25:	1987,26:	1988,27:	1989,28:	1990,29:	1991,30:	1992,31:	1993,32:	1994,33:	1995,34 :	1997}
df_meta['Year'] = [yeardict[s] for s in df_meta['SEAM']]


authors = df_meta['Author(s)'].values.astype(str)
authors = [s.replace('.','').replace('and',',').replace('&',',') for s in authors]
authors = [s.split(',')[0].replace(',','').strip().replace('  ', ' ') for s in authors]
authors = ['No Author' if s=='nan' else s for s in authors ]
authors

df_meta['First Author'] = authors

df_meta['SEAM EDX URL'] = r'https://edx.netl.doe.gov/dataset/seam-' + df_meta['SEAM'].apply(str)
#%%

# df_meta = df_meta[df_meta['SEAM'].isin([2,3])]


filename_stopwords = ["\'","\"", ".",",",":", "?" , "/"]

new_filepaths = []

for ID, row in df_meta.iterrows():
    relative_fp = row['Filepath']

    print("Processing paper " + str(ID) + " " + str(row['Title']))

    if relative_fp == relative_fp:
        input_fp = os.path.join(seams_pdf_folder, relative_fp)
        input_fp =  "\\\\?\\" + input_fp.replace('/', '\\')

        rel_filename_out =  'ID' + str(ID) + '_' + str(row['Year']) + '_' + row['First Author'] + '_'+ row['Title']

        for stopword in filename_stopwords:
            rel_filename_out = rel_filename_out.replace(stopword, "")

        max_rel_fp_length = 260 - len(old_seams_pdf_folder) - 30

        rel_filename_out = rel_filename_out[:max_rel_fp_length]

        rel_filename_out = rel_filename_out + '.pdf'

        #Define and make output folders
        rel_folder_out =  'SEAM' + str(row['SEAM'])
        output_folder_nocover = os.path.join(output_folder, 'NoCovers', rel_folder_out)
        if not os.path.exists(output_folder_nocover): os.makedirs(output_folder_nocover)
        output_fp_nocover = os.path.join(output_folder_nocover, rel_filename_out)

        output_folder_withcover = os.path.join(output_folder, 'WithCovers', rel_folder_out)
        if not os.path.exists(output_folder_withcover): os.makedirs(output_folder_withcover)
        output_fp_withcover = os.path.join(output_folder_withcover, rel_filename_out)

        temp_folder = r'C:\Users\aspit\Desktop\test seams\Temp'

        cmd_str = r'C:\Users\aspit\cpdf.exe -squeeze "' + input_fp + r'" -o "' + output_fp_nocover + r'"'
        os.system(cmd_str)


        print("Adding Cover")
        add_cover_header(output_fp_nocover, output_fp_withcover, row, temp_folder)

        # shutil.copy(input_fp, output_fp)

        df_meta.loc[ID,'Filepath'] = os.path.join(rel_folder_out, rel_filename_out)
# %%

# df_meta['Filepath'] = new_filepaths

meta_output_fp = os.path.join(output_folder, 'SEAMs_metadata.csv')
df_meta.to_csv(meta_output_fp)
# %%
