from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

embed = HuggingFaceBgeEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs = {'device':'cpu'}
)
text_splitter = RecursiveCharacterTextSplitter (
    chunk_size=1024,
    chunk_overlap=512,
)
df = pd.read_csv('../dataset/resumes.csv')
loader = DataFrameLoader(df,page_content_column='resume')

documents = loader.load()
document_chunks = text_splitter.split_documents(documents)
vector_db = FAISS.from_documents(documents=document_chunks,embedding=embed)
vector_db.save_local('../vectordb/')
