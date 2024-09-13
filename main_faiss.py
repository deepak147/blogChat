from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter


if __name__ == "__main__":

    print("Loading")
    loader = PyPDFLoader("research_assistant_medium_blog.pdf")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    split_docs = text_splitter.split_documents(documents=docs)

    embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
    llm = ChatOllama(model="qwen2:1.5b")
    vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    print("Ingesting")
    vector_store.save_local("faiss_embeddings")

    vs = FAISS.load_local(
        "faiss_embeddings", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs = create_stuff_documents_chain(llm=llm, prompt=retrieval_prompt)
    qa_chain = create_retrieval_chain(
        retriever=vs.as_retriever(), combine_docs_chain=combine_docs
    )

    res = qa_chain.invoke({"input": "Give me a 10 line brief of the blog"})
    print(res)