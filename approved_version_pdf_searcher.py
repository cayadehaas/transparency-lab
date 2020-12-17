from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urljoin
from tqdm import tqdm
from retry import retry


def get_sitemap():
    with open("firms_part_1.txt", "r") as file:
        urls = file.readlines()
        sitemap_list = []
        for url in urls:
            url = url.strip('\n')
            if requests.get(url + 'sitemap_index.xml').status_code == 200:
                print('1')
                sitemap_list.append([requests.get(url + 'sitemap_index.xml').text])
            elif requests.get(url + 'sitemap.xml').status_code == 200:
                print('2')
                sitemap_list.append([requests.get(url + 'sitemap.xml').text])
            elif requests.get(url + 'sitemap-index.xml').status_code == 200:
                print('3')
                sitemap_list.append([requests.get(url + 'sitemap-index.xml').text])
            elif requests.get(url + 'special-pages/google-site-map').status_code == 200:
                print('4')
                sitemap_list.append([requests.get(url + 'special-pages/google-site-map').text])
            elif requests.get(url + 'sitemap-list').status_code == 200:
                print('5')
                sitemap_list.append([requests.get(url + 'sitemap-list').text])
            elif requests.get(url + 'sitemap.aspx').status_code == 200:
                print('6')
                sitemap_list.append([requests.get(url + 'sitemap.aspx').text])
            elif requests.get(url + 'sitemapGEHC.xml').status_code == 200:
                print('7')
                sitemap_list.append([requests.get(url + 'sitemapGEHC.xml').text])
            elif requests.get(url + 'sitemap_en.xml').status_code == 200:
                print('8')
                sitemap_list.append([requests.get(url + 'sitemap_en.xml').text])
            elif requests.get(url + 'sitemap').status_code == 200:
                print('9')
                sitemap_list.append([requests.get(url + 'sitemap').text])
            else:
                print('Unable to fetch sitemap: %s.' % url)


        listoflists = []
        for sitemap in sitemap_list:
            soup = BeautifulSoup(str(sitemap), 'lxml')

            xml_urls = []
            for loc in soup.findAll('loc'):
                xml_urls.append(loc.text)

            result = []
            while xml_urls:
                candidate = xml_urls.pop()

                if is_sub_sitemap(candidate):
                    try:
                        sub_sitemap = requests.get(candidate).text
                        for i in process_sitemap(sub_sitemap):
                            xml_urls.append(i)
                    except:
                        continue
                else:
                    result.append(candidate)

            listoflists.append(result)

        with open("firms_and_url.txt", "r") as r:
            name_and_urls = r.readlines()
            parent_folder_location = r'E:\test_pdfs'
            if not os.path.exists(parent_folder_location): os.mkdir(parent_folder_location)
            for index, urls_per_brand in enumerate(listoflists):
                for index_urls, url in enumerate(name_and_urls):
                    if index_urls == index:
                        directory = url.strip(" \n").split(",")[0]
                        path = os.path.join(parent_folder_location, directory)
                        if not os.path.exists(path): os.makedirs(path)

                for url_in_urls in tqdm(urls_per_brand):
                    print(url_in_urls)
                    print(path)
                    url = url_in_urls.strip('\n')
                    try:
                        response = requests.get(url)
                        try:
                            soup = BeautifulSoup(response.text, "html.parser")
                            for link in soup.select("a[href$='.pdf']"):
                                # Name the pdf files using the last portion of each link which are unique in this case
                                filename = os.path.join(path, link['href'].split('/')[-1])
                                filename = filename.strip('\n')
                                if not "cv" or "affidavit" in filename:
                                    with open(filename, 'wb') as file:
                                        if requests.get(urljoin(url, link['href'])).status_code == 200:
                                            file.write(requests.get(urljoin(url, link['href'])).content)
                                        else:
                                            pass
                                else:
                                    print(filename)
                        except TypeError:
                            continue
                    except OSError or RuntimeError:
                        if OSError:
                            print('re-establish connection')
                            retry(delay=3)
                        else:
                            continue


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


def main():
    get_sitemap()


if __name__ == '__main__':
    main()
