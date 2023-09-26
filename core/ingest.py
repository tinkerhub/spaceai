from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    UnstructuredURLLoader,
    TextLoader,
    GitLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import traceback
import dotenv
import json
import os

dotenv.load_dotenv("ops/.env")

DATA_PATH = "data/"
DB_FAISS_PATH = os.environ.get('DB_FAISS_PATH')

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

with open("data/dummy-lp.json") as f:
    learning_paths = json.load(f)

with open("data/urls.txt") as f:
    tinkerhub_urls = f.readlines()
    tinkerhub_urls = [url.strip() for url in tinkerhub_urls]


def get_tinkerhub_docs(tinkerhub_urls):
    """
    This function gets the docs from
    the tinkerhub web pages.
    """
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
    """
    This function gets the docs from
    the txt files in data folder.
    """
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
    """
    This function gets the docs from
    the pdf files in data folder.
    """
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

def get_pdf_doc(filename: str):
    """
    This function gets the doc from
    the pdf file name passed in data
    folder
    """
    try:
        loader = PyPDFLoader(
            file_path=f"{DATA_PATH}{filename}"
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

def get_git_docs(url: str):
    """
    This function gets the docs from
    a git repo and save it under /repos/.
    """
    name = url.split("/")[-1]
    try:
        loader = GitLoader(
            clone_url=url,
            repo_path=f"repos/{name}",
            branch="main",
            file_filter=lambda file_path: file_path.endswith(".md"),
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

def get_learning_path_docs(learning_paths):
    docs = []
    for topic, url in learning_paths.items():
        docs.append(
            f"This is the maker station learning path for {topic} by TinkerHub. {url}"
        )
    return docs
        

def create_vector_db(docs):
    """
    This function creates a vector 
    database from docs from various
    sources.
    """
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def update_vector_db(
        docs: list, 
        vector_path: str = DB_FAISS_PATH, 
        emebddings: OpenAIEmbeddings = embeddings
):
    """
    This function updates the vector db
    """
    db = FAISS.load_local(vector_path, emebddings)
    print(docs)
    db.add_documents(docs)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    docs = get_tinkerhub_docs(tinkerhub_urls)
    docs.extend(get_pdf_docs())
    docs.extend(get_txt_docs())
    db = create_vector_db(docs)
    db.add_texts(get_learning_path_docs(learning_paths))
