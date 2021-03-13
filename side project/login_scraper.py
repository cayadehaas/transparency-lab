from bs4 import BeautifulSoup
from csv import reader
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import re
import pandas as pd
import csv
from time import sleep

USERNAME = "Marialisa@vandepoll.com"
PASSWORD = "4L@pX!je9TuVs7R"

LOGIN_URL = "https://www.artprice.com/identity"
URL = "https://www.artprice.com/artist/18956/agnes-martin/lots/pasts/1/painting"

estimate_1 = "€ [0-9]+,[0-9]+,[0-9]+ - € [0-9]+,[0-9]+,[0-9]+"  # € 1,483,619 - € 2,060,582
estimate_2 = "€ [0-9]+,[0-9]+ - € [0-9]+,[0-9]+"                #  € 16,909 - € 33,819 // € 16,90 - € 33,81

hammer_price_1 = "€ [0-9]+,[0-9]+,[0-9]+"   # € 1,648,466
hammer_price_2 = "€ [0-9]+,[0-9]+"          # € 212,195

date = '[0-9][0-9] [A-Z][a-z][a-z] [0-9]+' # 07 Dec 2020


def main():
    with open('Map_artist_links.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        with open('test_information_paintings.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ARTIST", 'TITLE', 'DATE', 'HAMMER PRICE', 'ESTIMATE'])
            payload = {
            "login": USERNAME,
            "pass": PASSWORD,
            }
            driver = webdriver.Chrome(executable_path="/Users/cayadehaas/PycharmProjects/Transparency-Lab/chromedriver")
            driver.get(LOGIN_URL)
            driver.find_element_by_id("login").send_keys(USERNAME)
            driver.find_element_by_id("pass").send_keys(PASSWORD)
            # driver.find_elements_by_class_name("sln-submit-login btn btn-primary").click()
            breakpoint()
            for artist, url in csv_reader:
                link = "".join(url)
                driver.get(link)
                while True:
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    ARTIST = artist

                    for painting in soup.find_all("a", {"class": "sln_lot_show"}):
                        for title in painting.find("div", {"class": "lot-datas-title"}):
                            TITLE = title.replace('\n', '').strip()

                        data = painting.find_all("div", {"class": "lot-datas-block"})
                        DATE = re.compile("(%s)" % (date)).findall(str(data))
                        DATE = (", ".join(DATE))
                        ESTIMATE = re.compile("(%s|%s)" % (estimate_1, estimate_2)).findall(str(data))
                        ESTIMATE = (", ".join(ESTIMATE))

                        # get the hammer price // not sold // not listed
                        if re.compile("(%s|%s)" % (hammer_price_1, hammer_price_2)).findall(str(data)) == 4:
                            HAMMER_PRICE = re.compile("(%s|%s)" % (hammer_price_1, hammer_price_2)).findall(str(data))
                            HAMMER_PRICE = HAMMER_PRICE[-1]

                            print([ARTIST, TITLE, DATE, HAMMER_PRICE, ESTIMATE])
                            writer.writerow([ARTIST, TITLE, DATE, HAMMER_PRICE, ESTIMATE])

                        else:
                            if re.compile("Not sold").findall(str(data)) != 0:
                                NOT_HAMMER_PRICE = 'Not sold'
                            elif re.compile("Not listed").findall(str(data)) != 0:
                                NOT_HAMMER_PRICE = 'Not listed'
                            else:
                                NOT_HAMMER_PRICE = "-"

                            print([ARTIST, TITLE, DATE, NOT_HAMMER_PRICE, ESTIMATE])
                            writer.writerow([ARTIST, TITLE, DATE, NOT_HAMMER_PRICE, ESTIMATE])
                    sleep(10)
                    try:
                        driver.execute_script("return arguments[0].scrollIntoView(true);",
                                              WebDriverWait(driver, 20).until(
                                                  EC.element_to_be_clickable((By.XPATH, "//li[@class='next_page']/a"))))
                        driver.find_element_by_xpath("//li[@class='next_page']/a").click()

                    except (TimeoutException, WebDriverException) as e:
                        print("Last page reached")
                        break

        driver.quit()

        read_file = pd.read_csv(r'test_information_paintings.csv')
        read_file.to_excel(r'test_information_paintings.xlsx', index=None, header=True)


if __name__ == '__main__':
    main()
