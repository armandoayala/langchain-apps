from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

embeddings = OpenAIEmbeddings()


def init_vector_db():
    loader = PyPDFLoader(file_path="data/react_llm.pdf")
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    texts = text_splitter.split_documents(document)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("vectordb/faiss_index_start")


def query_ask(query):
    vectorstore_db = FAISS.load_local("vectordb/faiss_index_start", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vectorstore_db.as_retriever()
    )

    answer = qa.run(query)
    return answer


def run():
    answer = query_ask("Give me the gist of ReAct in 3 sentences")
    print(answer)
