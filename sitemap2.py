from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urljoin
f = open("xmlfiles.txt", "a")


def get_sitemap():
    with open("get_consultancy_firms.txt", "r") as file:
        urls = file.readlines()
        for url in urls:
            url = url.strip('\n')
            if requests.get(url + 'sitemap').status_code == 200:
                return requests.get(url + 'sitemap')
            elif requests.get(url + 'sitemap.xml').status_code == 200:
                return requests.get(url + 'sitemap.xml')
            elif requests.get(url + 'sitemap_index.xml').status_code == 200:
                return requests.get(url + 'sitemap_index.xml')
            elif requests.get(url + 'sitemap-index.xml').status_code == 200:
                return requests.get(url + 'sitemap-index.xml')
            elif requests.get(url + 'special-pages/google-site-map').status_code == 200:
                return requests.get(url + 'special-pages/google-site-map')
            elif requests.get(url + 'sitemap-list').status_code == 200:
                return requests.get(url + 'sitemap-list')
            elif requests.get(url + 'sitemap.aspx').status_code == 200:
                return requests.get(url + 'sitemap.aspx')
            elif requests.get(url + 'sitemapGEHC.xml').status_code == 200:
                return requests.get(url + 'sitemapGEHC.xml')
            elif requests.get(url + 'sitemap_en.xml').status_code == 200:
                return requests.get(url + 'sitemap_en.xml')
            else:
                print('Unable to fetch sitemap: %s.' % url)


def process_sitemap(s):
    """retrieve text from all <loc>"""
    soup = BeautifulSoup(s, 'lxml')
    result = []
    for loc in soup.findAll('loc'):
        result.append(loc.text)

    return result


def is_sub_sitemap(s):
    """checks for xml files"""
    if s.endswith('.xml') and 'sitemap' in s:
        return True
    else:
        return False


def parse_sitemap(s):
    sitemap = process_sitemap(s)
    result = []

    while sitemap:
        candidate = sitemap.pop()

        if is_sub_sitemap(candidate):
            sub_sitemap = get_sitemap()
            for i in process_sitemap(sub_sitemap):
                sitemap.append(i)
        else:
            result.append(candidate)

    return result

#
# def pdf_searcher():
#     """get request on all urls, checks for pdf files and add them to folder"""
#     folder_location = r'E:\web_scraping'
#     if not os.path.exists(folder_location): os.mkdir(folder_location)
#
#     with open("xmlfiles.txt", "r") as file:
#         urls = file.readlines()
#         for url in urls:
#             url = url.strip('\n')
#             response = requests.get(url)
#             print(response)
#             soup = BeautifulSoup(response.text, "html.parser")
#             for link in soup.select("a[href$='.pdf']"):
#                 # Name the pdf files using the last portion of each link which are unique in this case
#                 filename = os.path.join(folder_location, link['href'].split('/')[-1])
#                 with open(filename, 'wb') as file:
#                     file.write(requests.get(urljoin(url, link['href'])).content)


def main():
    sitemap = get_sitemap()
    url = '\n'.join(parse_sitemap(sitemap))
    # pdf_searcher()


if __name__ == '__main__':
    main()
