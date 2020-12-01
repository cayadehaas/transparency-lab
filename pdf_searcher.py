from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urljoin
from tqdm import tqdm
from retry import retry


def get_sitemap():
    with open("firms_part_1.txt", "r") as file:
        urls = file.readlines()
        sitemap = ''
        for url in urls:
            url = url.strip('\n')

            if requests.get(url + 'sitemap_index.xml').status_code == 200:
                print('1')
                sitemap += requests.get(url + 'sitemap_index.xml').text
            elif requests.get(url + 'sitemap.xml').status_code == 200:
                print('2')
                sitemap += requests.get(url + 'sitemap.xml').text

            elif requests.get(url + 'sitemap-index.xml').status_code == 200:
                print('3')
                sitemap += requests.get(url + 'sitemap-index.xml').text
            elif requests.get(url + 'special-pages/google-site-map').status_code == 200:
                print('4')
                sitemap += requests.get(url + 'special-pages/google-site-map').text
            elif requests.get(url + 'sitemap-list').status_code == 200:
                print('5')
                sitemap += requests.get(url + 'sitemap-list').text
            elif requests.get(url + 'sitemap.aspx').status_code == 200:
                print('6')
                sitemap += requests.get(url + 'sitemap.aspx').text
            elif requests.get(url + 'sitemapGEHC.xml').status_code == 200:
                print('7')
                sitemap += requests.get(url + 'sitemapGEHC.xml').text
            elif requests.get(url + 'sitemap_en.xml').status_code == 200:
                print('8')
                sitemap += requests.get(url + 'sitemap_en.xml').text
            elif requests.get(url + 'sitemap').status_code == 200:
                print('9')
                sitemap += requests.get(url + 'sitemap').text
            else:
                print('Unable to fetch sitemap: %s.' % url)

    return sitemap


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
            sub_sitemap = requests.get(candidate).text
            for i in process_sitemap(sub_sitemap):
                sitemap.append(i)
        else:
            result.append(candidate)
    len(result)
    return result


def pdf_searcher(urls):
    """get request on all urls, checks for pdf files and add them to folder"""
    folder_location = r'E:\pdfs_consultancy_firms'
    if not os.path.exists(folder_location): os.mkdir(folder_location)

    for url in tqdm(urls):
        url = url.strip('\n')
        try:
            response = requests.get(url)
            try:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.select("a[href$='.pdf']"):
                    # Name the pdf files using the last portion of each link which are unique in this case
                    filename = os.path.join(folder_location, link['href'].split('/')[-1])
                    filename = filename.strip('\n')
                    with open(filename, 'wb') as file:
                        if requests.get(urljoin(url, link['href'])).status_code == 200:
                            file.write(requests.get(urljoin(url, link['href'])).content)
                        else:
                            pass
            except TypeError:
                continue
        except OSError or RuntimeError:
            if OSError:
                print('re-establish connection')
                retry(delay=3)
            else:
                continue


def main():
    sitemap = get_sitemap()
    urls = parse_sitemap(sitemap)
    pdf_searcher(urls)


if __name__ == '__main__':
    main()
