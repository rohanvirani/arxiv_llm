from urllib.request import urlopen
from bs4 import BeautifulSoup
from llama_index import Document, GPTSimpleVectorIndex, ServiceContext
from llama_index.llm_predictor import StableLMPredictor
from typing_extensions import Protocol

url = 'http://export.arxiv.org/api/query?search_query=au:"Percy%20Liang"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
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
response = index.query("Explain these abstracts to a venture capitalist?")
print(response)

