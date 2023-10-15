import os
import pinecone
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
embeddings = OpenAIEmbeddings()
INDEX_NAME = "dev-labs-index"


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf8"
    )
    raw_documents = loader.load()
    print(f" loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")


def run_app():
    print("Start App")
    ingest_docs()
