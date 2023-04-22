import streamlit as st
import pandas as pd
from urllib.request import urlopen
from llama_index import Document, GPTSimpleVectorIndex, ServiceContext
from llama_index.llm_predictor import StableLMPredictor
from bs4 import BeautifulSoup
import os

# i want to enter someone's name and get their information and ask questions about their work

# user enters name
# fetch arxiv papers
# index over papers using llama index
# prompt and learn about papers and coauthors 
# serve up on front end 


def gen_results(name, query):
    name_split = name.split()
    url = 'http://export.arxiv.org/api/query?search_query=au:"' + name_split[0] + '%20' + name_split[1] + '"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
    with urlopen(url) as response:
        data = response.read()

    soup = BeautifulSoup(data, 'xml')
    article_info = soup.find_all('entry')

    abstract_list = []

    for i in article_info:
        abstract_list.append(i.summary.text)
    
    stablelm_predictor = StableLMPredictor()
    service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm_predictor=stablelm_predictor)
    documents = [Document(t) for t in abstract_list]
    index = GPTSimpleVectorIndex.from_documents(documents,service_context)
    response = index.query("What did the author do growing up?")
    return response

title = st.title("Amplify Academic Search")
name = st.text_input(label="Enter Researcher Name",value="Percy Liang")
query = st.text_input(label="Enter Question",value="Explain the following documents so that a venture capitalist can understand them")
if st.button('Generate Results'):
    st.write(gen_results(name, query))




