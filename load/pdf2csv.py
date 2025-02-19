import PyPDF2
import glob 
import os 
import re
import sys
import pandas as pd

CSV_PATH = '../dataset/'

def convert_csv(pdf_files):
    resume_dict = {'ids':[],'resume':[]}
    count = 0
    extractedtext = ""
    ids = 0
    for pdf_file in pdf_files:
        #print(pdf_file)
        #pdfFileObj = open(pdf_file.read(),'rb')               
        pdfReader = PyPDF2.PdfReader(pdf_file)   
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
    return df
        
