import streamlit as st
import pandas as pd
import sys, os

from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from .Chatbot import Chatbot
from .retriever import QueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_modal import Modal
sys.path.append('../')
from io import StringIO
from load.pdf2csv import *
from load.vectorize import *
import openai
import PyPDF2


introduction_message = """
Welcome to ResumeRAG – Your AI-Powered Resume Screening Assistant!

ResumeRAG is an advanced AI-driven system designed to simplify and enhance the hiring process. As the sole developer, I’ve built this tool to help hiring managers efficiently evaluate candidates with the power of OpenAI’s API and retrieval-augmented generation (RAG) technology.

🔍 How ResumeRAG Works:  \n

✅ Extracts relevant resumes using cutting-edge semantic search. \n
✅ Compares candidates based on job requirements and key qualifications.  \n
✅ Delivers AI-powered insights to support data-driven hiring decisions.  \n

No more manual filtering—let ResumeRAG intelligently match the best candidates for your role! 🚀

Simply enter a job description or candidate IDs to get started..
"""
st.title('ResumeRAG')

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv('./dataset/resumes.csv')
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=introduction_message)]
if "embedder" not in st.session_state:
    st.session_state.embedder = HuggingFaceBgeEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs = {'device':'cpu'}
)
if "retreiver" not in st.session_state:
    vectordb = FAISS.load_local(folder_path='./vectordb/',embeddings= st.session_state.embedder,   allow_dangerous_deserialization=True)
    st.session_state.retreiver = QueryRetriever(vectordb,st.session_state['df'])


def check_api_key(api_key:str):
    if api_key == None:
        st.error("They key field is empty")
        return False
    elif api_key != None:
        client = openai.OpenAI(api_key=api_key)
        try:
            client.models.list()
        except openai.AuthenticationError:
            return False
        else:
            return True


def upload_file_csv():
    modal = Modal(key="Demo Key",title="File error",max_width=400)
    if st.session_state.uploaded_file_csv != None:
        try:
            df = pd.read_csv(st.session_state.uploaded_file_csv)
        except Exception as error:
            with modal.container():
                st.write('The uploaded file is incorrect. Check the file for the following error.')
                st.error(error)
        else:
            if "resume" not in df.columns or "ids" not in df.columns:
                with modal.container():
                    st.error('The dataset should contain the columns resume and ids')
            else:
                with st.spinner('Uploading data and vectorizing it...'):
                    vectordb = store_vectordb(df,'resume',st.session_state.embedder)
                    st.session_state.retreiver = QueryRetriever(vectordb,df)
    else:
        st.error('Upload a csv file to continue')
def upload_file_pdfs():
    modal = Modal(key="Demo Key",title="File error",max_width=400)
    if st.session_state.uploaded_file_pdfs != None:
        try:
            resume_dict = {'ids':[],'resume':[]}
            count = 0
            extractedtext = ""
            ids = 0
            pdf_files = st.session_state.uploaded_file_pdfs
            for pdf_file in pdf_files:     
                pdfReader = PyPDF2.PdfReader(pdf_file)   
                num_pages = len(pdfReader.pages)   
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
        except Exception as error:
            with modal.container():
                st.write('The uploaded file is incorrect. Check the file for the following error.')
                st.error(error)
        else:
            if "resume" not in df.columns or "ids" not in df.columns:
                with modal.container():
                    st.error('The dataset should contain the columns resume and ids')
            else:
                with st.spinner('Uploading data and vectorizing it...'):
                    vectordb = store_vectordb(df,'resume',st.session_state.embedder)
                    st.session_state.retreiver = QueryRetriever(vectordb,df)
    else:
        st.error('Upload the pdf files to continue')
            
def check_model_name(api_key:str,model_name:str):
        client = openai.OpenAI(api_key=api_key)
        model_list = [model.id for model in client.models.list()]
        if model_name not in model_list:
            return False
        else:
            return True
def clear_message():
    st.session_state.chat_history = {AIMessage(content=introduction_message)}

user_query = st.chat_input('Message ResumeRAG...')

with st.sidebar:
    st.text_input(label='Type your OpenAI api key',key='api_key',type='password')
    st.text_input('Model name',"gpt-4o-mini",key='model_name')
    st.selectbox('RAG method',options=['GeneralRAG','RAGfusion'],key='rag_method',placeholder='GeneralRAG')
    st.file_uploader('Upload Resumes in pdfs',key='uploaded_file_pdfs',type="pdf",accept_multiple_files=True,on_change=upload_file_pdfs)
    st.file_uploader('Upload Resumes in csv',key='uploaded_file_csv',on_change=upload_file_csv)
    st.divider()
    st.button('Clear conversation')


for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message('ai'):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message('human'):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])

if not st.session_state.api_key:
    st.info("Please enter the OpenAI API key to continue.")
    st.stop()
if not check_api_key(st.session_state.api_key):
    st.error("The entered API key does not seem to be right. Please enter a valid API key")
    st.stop()
if not check_model_name(st.session_state.api_key,st.session_state.model_name):
    st.error("The entered model GPT name seems to be incorrect. Please enter a valid GPT name. For more information check https://platform.openai.com/docs/models")
    st.stop()


    

llm = Chatbot(
    api=st.session_state.api_key,
    model = st.session_state.model_name,
)
retreiver = st.session_state.retreiver
if user_query != "" and user_query is not None:
    with st.chat_message("human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("ai"):
        with st.spinner('Generating answers...'):
            document_list = retreiver.retreive_resume(user_query,llm,st.session_state.rag_method)
            query_type = retreiver.meta_data['query_type']
            stream_message = llm.generate_message_stream(user_query,document_list,[],query_type)
        response = st.write_stream(stream_message)
        st.write_stream(stream_message)
        st.session_state.chat_history.append(AIMessage(content=response))

