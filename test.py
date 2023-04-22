from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'http://export.arxiv.org/api/query?search_query=au:"Percy%20Liang"&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending'
with urlopen(url) as response:
    data = response.read()

soup = BeautifulSoup(data, 'xml')
article_info = soup.find_all('entry')
for i in article_info:
    print(i.summary.text)