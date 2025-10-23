import streamlit as st
from PyPDF2 import PdfReader 
import os
import openai 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain

#upload pdf file
st.header("FM's Tiny Chatbot\n")
st.header("Upload your document and ask any question related to that document")
with st.sidebar :
    st.title("Open PDF Document")
    file = st.file_uploader("Uplodad PDF file",type="pdf")

#check the version
#version = "langchain version:", getattr(langchain, "__version__", "unknown")
#st.write("Lanchain version ", version )
#available_version = dir(getattr(langchain, "chains", None))
#st.write("Lanchain chains ", available_version )

#create chunks
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
            text+=page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len    
    )
    chunk= text_splitter.split_text(text)
    #Generating Embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #generating vecotrs FAISS
    vector_store = FAISS.from_texts(chunk,embeddings)
    #get user question
    user_input= st.text_input("Enter your query")
    
    if user_input is not None:
        
        #do similarity search
        match = vector_store.similarity_search(user_input) 
       
    openai.api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    context = "\n\n---\n\n".join([d.page_content for d in match[:6]])
    
    messages = [
    {"role": "system", "content": "You are an expert assistant. Use ONLY the provided excerpts to answer the question."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
    ]

    resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = messages,
    temperature=0.0,
    max_tokens=800,
    )
    response = resp["choices"][0]["message"]["content"].strip()
    st.markdown("**Response:**")

    st.write(response)

