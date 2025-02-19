from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd


text_splitter = RecursiveCharacterTextSplitter (
    chunk_size=1024,
    chunk_overlap=512,
)
#df = pd.read_csv('../dataset/resumes.csv')
def store_vectordb(df,content_column,embed_model):
    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=1024,
        chunk_overlap=512,
    )
    loader = DataFrameLoader(df,page_content_column=content_column)

    documents = loader.load()
    document_chunks = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(documents=document_chunks,embedding=embed_model)
    vector_db.save_local('../vectordb/')
    return vector_db
