from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    UnstructuredURLLoader,
    TextLoader
    
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import traceback
import dotenv
import os

dotenv.load_dotenv("ops/.env")
DATA_PATH = "data/"
DB_FAISS_PATH = os.environ.get('DB_FAISS_PATH')

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

tinkerhub_urls = ['https://www.tinkerhub.org/',
 'https://www.tinkerhub.org/learn',
 'https://www.tinkerhub.org/career-initiative',
 'https://www.tinkerhub.org/tinkerspace']


def get_tinkerhub_docs(tinkerhub_urls):
    docs = []
    for url in tinkerhub_urls:
        loader = UnstructuredURLLoader([url])
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_documents(documents)
        docs.extend(texts)
    return docs

def get_txt_docs():
    try:
        loader = DirectoryLoader(
            DATA_PATH, 
            glob='*.txt', 
            loader_cls=TextLoader
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)
    except Exception as e:
        print(traceback.format_exc())
        docs = []
    return docs


def get_pdf_docs():
    try:
        loader = DirectoryLoader(
            DATA_PATH, 
            glob='*.pdf', 
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)
    except Exception:
        print(traceback.format_exc())
        docs = []
    return docs



def create_vector_db(docs):
    """
    This function creates a vector 
    database from a directory of PDFs.
    """
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    docs = get_tinkerhub_docs(tinkerhub_urls)
    docs.extend(get_pdf_docs())
    docs.extend(get_txt_docs())
    create_vector_db(docs)
