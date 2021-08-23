from bs4 import BeautifulSoup
from selenium import webdriver


def get_text_from_url():
    """
        This script tries to open the url one by one and extract the text from the web page
        and places text from different tags in different variables
        To run this script, download chromedriver for your OS

    """
    with open('article_urls.txt') as f:  # place web page links in txt file
        driver = webdriver.Chrome(executable_path="/Users/cayadehaas/PycharmProjects/Transparency-Lab/chromedriver 2")  # change this to correct PATH
        urls = f.readlines()
        for url in urls:
            print(url)  # show web page link

            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            article_text_p = ''
            article_text_h2 = ''
            article_text_h3 = ''

            text_p_tags = soup.find("div").findAll('p')
            for element in text_p_tags:
                article_text_p += '\n' + ''.join(element.findAll(text=True))

            text_h2_tags = soup.find("div").findAll('h2')
            for element in text_h2_tags:
                article_text_h2 += '\n' + ''.join(element.findAll(text=True))

            text_h3_tags = soup.find("div").findAll('h3')
            for element in text_h3_tags:
                article_text_h3 += '\n' + ''.join(element.findAll(text=True))

            print('<p> tags text:', article_text_p)
            print('<h2> tags text:', article_text_h2)
            print('<h3> tags text:', article_text_h3)


def main():
    get_text_from_url()


if __name__ == '__main__':
    main()

