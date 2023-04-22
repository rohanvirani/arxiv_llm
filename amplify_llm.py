import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import urllib
import feedparser
from llama_index import Document, GPTSimpleVectorIndex

# i want to enter someone's name and get their information and ask questions about their work

# user enters name
# fetch arxiv papers
# index over papers using llama index
# prompt and learn about papers and coauthors 
# serve up on front end 

def gen_results(name, query):
    name_split = name.split()
    url = 'http://export.arxiv.org/api/query?search_query=au:"' + name_split[0] + '%20' + name_split[1] + '"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
    data = urllib.urlopen(url).read()
    feed = feedparser.parse(data)

    abstract_list = []

    for entry in feed.entries:
        abstract_list.append(entry.summary)
    documents = [Document(t) for t in abstract_list]
    index = GPTSimpleVectorIndex.from_documents(documents)
    response = index.query("What did the author do growing up?")
    return response

title = st.title("Amplify Academic Search")
name = st.text_input(label="Enter Researcher Name",value="Percy Liang")
query = st.text_input(label="Enter Question",value="Explain the following documents so that a venture capitalist can understand them")
if st.button('Generate Results'):
    st.write(gen_results(name, query))




