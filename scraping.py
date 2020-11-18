from bs4 import BeautifulSoup
import requests
import re

base_url = "https://en.m.wikipedia.org/"
wikipedia_url = "https://en.wikipedia.org/wiki/List_of_management_consulting_firms"
html_content = requests.get(wikipedia_url).text
f = open("wikipedia_pages_consultancy_firms.txt", "a")

soup = BeautifulSoup(html_content, "html.parser")
consultancy_firm_links = soup.find_all("ul")

for a_tag in consultancy_firm_links[0].find_all('a', href=True):
    page_links = base_url + a_tag['href']
    f.write(page_links + "\n")
f.close()


with open("wikipedia_pages_consultancy_firms.txt", "r") as file:
    pages = file.readlines()
    pages = [x.strip() for x in pages]
print(pages)




# url = "https://www.myconsultingoffer.org/list-top-management-firms/"
# pagination = soup.find("div", attrs={"class": "pagination"})
# print(pagination)
# page_count_links = soup.find_all("a", datapageid=re.compile(r".*"))




# consultancy_table = soup.find_all("table", attrs={"id": "customers"})
# for tr in consultancy_table[-1].find_all('tr'):
#     td = tr.find('td')
#     if td:
#         print(td)


