from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import dotenv
import os

dotenv.load_dotenv("ops/.env")
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"



def create_vector_db():
    loader = DirectoryLoader(
        DATA_PATH, 
        glob='*.pdf', 
        loader_cls=PyPDFLoader()
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        os.getenv('OPENAI_API_KEY'),
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
