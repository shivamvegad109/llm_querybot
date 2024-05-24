import os
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

# Load the LLama2 model outside the function
llm = Ollama(model="llama2")

def getLLamaresponse_with_vector_prompt(input_text, no_words, refrence):
    # Prompt Template
    template = """
        Write a blog about "{input_text}"
        with approximately {no_words} words and
        Refrence like "{refrence}".
        """
    prompt = PromptTemplate(input_variables=["input_text", "no_words", "refrence"], template=template)

    # Generate the response from the LLama2 model
    response = llm.invoke(prompt.format(input_text=input_text, no_words=no_words, refrence = refrence))
    return response

def getLLamaresponse(input_text, no_words):
    # Prompt Template
    template = """
        Write a blog about "{input_text}"
        with approximately {no_words} words.
        """
    prompt = PromptTemplate(input_variables=["input_text", "no_words"], template=template)

    # Generate the response from the LLama2 model
    response = llm.invoke(prompt.format(input_text=input_text, no_words=no_words))
    return response


def doc_save_local(file, embedding):
    loader = TextLoader(file)
    documents = loader.load()

    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 0,
        length_function = len,
    )

    documents = test_splitter.split_documents(documents)

    library = FAISS.from_documents(documents, embedding)

    library.save_local('faiss_index_metallica')


def load_model_to_local(embedding, retriever_query):

    metallica_saved = FAISS.load_local("faiss_index_metallica",embedding, allow_dangerous_deserialization=True)
    qa_saved = RetrievalQA.from_chain_type(llm = OpenAI(), chain_type="stuff", retriever=metallica_saved.as_retriever())
    results = qa_saved.invoke(retriever_query)
    return results


def visualize_data(embedding):
    metallica_saved = FAISS.load_local("faiss_index_metallica",embedding, allow_dangerous_deserialization=True)
    ids = []
    page_contents = []
    sources = []
    store = metallica_saved.docstore._dict

    for key, value in store.items():
        ids.append(key)
        page_contents.append(value.page_content)
        sources.append(value.metadata['source'])

    model_data = pd.DataFrame({
        'ID': ids,
        'Page Content': page_contents,
        'Source': sources
    })
    
    return model_data


def delete_document(source, embedding):
    metallica_saved = FAISS.load_local("faiss_index_metallica",embedding, allow_dangerous_deserialization=True)
    data = visualize_data(embedding)
    chunks_list = data.loc[data['Source']==source]['ID'].tolist()
    if chunks_list:
        metallica_saved.delete(chunks_list)
    metallica_saved.save_local('faiss_index_metallica')


# def delete_document(source, embedding):
#     metallica_saved = FAISS.load_local("faiss_index_metallica",embedding, allow_dangerous_deserialization=True)
#     store = metallica_saved.docstore._dict
#     for document_id, document_info in store.items():
#         print(document_id)
#         print(document_info.metadata['source'])
#         del document_info.metadata['source']
#         metallica_saved.docstore._dict = store
#         metallica_saved.save_local('faiss_index_metallica')