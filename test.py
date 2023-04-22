from urllib.request import urlopen
from bs4 import BeautifulSoup
from llama_index import Document, GPTSimpleVectorIndex

url = 'http://export.arxiv.org/api/query?search_query=au:"Percy%20Liang"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
with urlopen(url) as response:
    data = response.read()

soup = BeautifulSoup(data, 'xml')
article_info = soup.find_all('entry')

abstract_list = []

for i in article_info:
    abstract_list.append(i.summary.text)
documents = [Document(t) for t in abstract_list]
index = GPTSimpleVectorIndex.from_documents(documents)
response = index.query("What did the author do growing up?")
print(response)

