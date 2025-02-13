import PyPDF2
import glob 
import os 
import re
import sys
import pandas as pd

CSV_PATH = '../dataset/'

dir_to_read = sys.argv[1] 
pdf_files = glob.glob(os.path.join(dir_to_read,'*.pdf'))
resume_dict = {'ids':[],'resume':[]}
count = 0
extractedtext = ""
ids = 0
for pdf_file in pdf_files:
    print(pdf_file)
    pdfFileObj = open(pdf_file,'rb')               
    pdfReader = PyPDF2.PdfReader(pdfFileObj)   
    num_pages = len(pdfReader.pages)   

    #print(num_pages)
    extractedtext = ""
    count = 0
    while count < num_pages:                       
        pageObj = pdfReader.pages[count]
        count +=1
        extractedtext += pageObj.extract_text()
    
    resume_dict['ids'].append(ids)
    resume_dict['resume'].append(extractedtext)
    ids += 1
df = pd.DataFrame(resume_dict,index=None)
df.to_csv(CSV_PATH + 'resumes.csv',index=False)
    
