import string
import PyPDF2
import random
import string
import requests

import streamlit as st

from langchain_cohere import ChatCohere
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_kb_from_img(file) : 

    with open('file.png' , 'wb') as upfile : upfile.write(file.getbuffer())

    link = upload_img()

    chat = ChatCohere(cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')

    prompt = f'''
    You are a Product Marketing Chatbot, You will be provided with an image, Your task is to cpationaize the image such that it is relevant to the image and the context of the image a
    
    Image Link : {link}
    '''

    messages = [HumanMessage(content = prompt)]
    response = chat.invoke(messages).content

    documents = [
        Document(
            page_content = response , 
            metadata = {
                'source' : 'image' , 
                'file_name' : file.name , 
                'image_link' : link ,
            }
        )
    ]

    return documents

def extract_kb_from_pdf(file) : 

    documents = []

    pdf = PyPDF2.PdfReader(file)
    
    text = ' '.join([
        pdf.pages[page_number].extract_text()
        for page_number 
        in range(len(pdf.pages))
    ])

    chunks = [
        text[index : index + 1024]
        for index
        in range(0 , len(text) , 1024)
    ]

    for chunk in chunks : 
        documents.append(Document(
            page_content = chunk , 
            metadata = {
                'source' : 'pdf' , 
                'file_name' : file.name
            }
        ))

    return documents

def upload_img() : 

    url = f"https://api.imgbb.com/1/upload?expiration=600&key=e3c6743b30cd5794d8321e50092683a3"
    
    with open('file.png' , 'rb') as file : 

        payload = {'image' : file.read()}
        response = requests.post(url, files=payload)
        
    if response.status_code == 200 : response = response.json()
    else : response = response.text

    link = response['data']['url']

    return link



    dropbox_destination_path = '/' + generate_random_string() + '.jpg'

    upload_image_to_dropbox('file.png' , dropbox_destination_path)

    public_url = create_shared_link(dropbox_destination_path)

    return public_url

file = st.file_uploader('Upload a PDF file' , type = ['pdf' , 'png' , 'jpg'])
query = st.text_input('Enter your query')

if st.button('Ask') : 

    if file.name.endswith('pdf') : documents = extract_kb_from_pdf(file)
    else : documents = extract_kb_from_img(file)

    embeddings = HuggingFaceEmbeddings(
        model_name = 'all-MiniLM-L6-v2'
    )

    vc = FAISS.from_documents(
        documents = documents , 
        embedding = embeddings
    )

    similar_docs = vc.similarity_search(query)

    context = ' '.join([
        doc.page_content
        for doc
        in similar_docs
    ])

    chat = ChatCohere(cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')

    prompt = '''
You are a Product Marketing Speicalist, you will be given query, your task is to answer the query with the best of your knowledge.

Context = {}

Query : {}
    '''

    prompt = prompt.format(context , query)

    messages = [HumanMessage(content = prompt)]

    st.write(chat.invoke(messages).content)
