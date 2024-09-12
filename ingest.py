import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if __name__ == "__main__":

    print("Loading doc")
    loader = PyPDFLoader("research_assistant_medium_blog.pdf")
    doc = loader.load()

    print("Splitting")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_text = splitter.split_documents(doc)
    print(f"Split into {len(split_text)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")

    print("Ingesting through embedding model")
    index_name = os.environ.get("INDEX_NAME")
    vector_store = PineconeVectorStore.from_documents(
        documents=split_text, embedding=embeddings, index_name=index_name
    )
    print("finish")
