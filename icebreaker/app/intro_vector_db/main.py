import os
import pinecone
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
embeddings = OpenAIEmbeddings()
index_name = "medium-blogs-index"


def delete_data_index():
    index = pinecone.Index("medium-blogs-index")
    # remove from namespace="Default"
    index.delete(delete_all=True)


def load_vector_db():
    loader = TextLoader("data/medium_vector_db.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    return docsearch


def query_ask(query):
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    answer = qa({"query": query})
    return answer


def run():
    answer = query_ask("What is a vector DB? Give me a 15 word answer for a begginer")
    print(answer)
