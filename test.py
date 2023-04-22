from urllib.request import urlopen
from bs4 import BeautifulSoup
from llama_index import Document, GPTSimpleVectorIndex, ServiceContext
from llama_index.llm_predictor import StableLMPredictor
from typing_extensions import Protocol
from langchain import HuggingFaceHub

url = 'http://export.arxiv.org/api/query?search_query=au:"Percy%20Liang"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
with urlopen(url) as response:
    data = response.read()

soup = BeautifulSoup(data, 'xml')
article_info = soup.find_all('entry')

abstract_list = []

for i in article_info:
    abstract_list.append(i.summary.text)
repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm_predictor=llm)
documents = [Document(t) for t in abstract_list]
index = GPTSimpleVectorIndex.from_documents(documents,service_context)
response = index.query("Explain these abstracts to a venture capitalist?")
print(response)

