import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":

    print("Retrieving")

    embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
    llm = ChatOllama(model="qwen2:1.5b")

    query = "What does qdrant do in an LLM setting?"

    index_name = os.environ["INDEX_NAME"]
    vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)

    template = """Use the following pieces of ocntext to answer the below given question. If you cannot come up with an answer, don't make up things. Let the answer be concise and within 3 lines. Complete the answer and say 'Thanks for asking!'. 
    
    {context}
    
    Question: {question}
    
    Helpful answer:"""
    custom_prompt = PromptTemplate.from_template(template=template)

    custom_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
    )

    res = custom_chain.invoke(query)

    print(res)