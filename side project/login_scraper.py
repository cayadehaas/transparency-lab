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
estimate_3 = "€ [0-9][0-9][0-9] - € [0-9][0-9][0-9]"             # € 406 - € 580
estimate_4 = "€ [0-9][0-9] - € [0-9][0-9]+"                      # 40 - 50
estimate_5 = "€ [0-9]+ - € [0-9]+,[0-9]+"                        # € 906 - € 1,580
estimate_6 = "€ [0-9]+,[0-9]+,[0-9]+"                           # € 1,483,619
estimate_7 = "€ [0-9]+"                                          # € 452
estimate_8 = "€ [0-9]+,[0-9]+"                                  # € 16,909
estimate_9 = "€ [0-9],[0-9]+"

hammer_price_1 = "€ [0-9]+,[0-9]+,[0-9]+"   # € 1,648,466
hammer_price_2 = "€ [0-9]+,[0-9]+"          # € 212,195
hammer_price_3 = "€ [0-9]+"                 # € 452

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
                        content = str(data)
                        DATE = re.compile("(%s)" % (date)).findall(str(data))
                        DATE = (", ".join(DATE))

                        # Estimate is € 1,483,619 - € 2,060,582, or  € 483,619 - 343,000, or € 483 - 343 or empty
                        if 'Estimate' in content:
                            if len(re.compile("(%s|%s|%s|%s|%s)" % (estimate_1, estimate_2, estimate_3, estimate_4, estimate_5)).findall(str(data))) != 0:
                                ESTIMATE = re.compile("(%s|%s|%s|%s|%s)" % (estimate_1, estimate_2, estimate_3, estimate_4, estimate_5)).findall(str(data))
                                ESTIMATE = (", ".join(ESTIMATE))

                            # Estimate is € 1,483,619, € 1,483, € 134 or empty
                            elif len(re.compile("(%s|%s|%s|%s)" % (estimate_6, estimate_7, estimate_8, estimate_9)).findall(str(data))) != 0:
                                ESTIMATE = re.compile("(%s|%s|%s|%s)" % (estimate_6, estimate_7, estimate_8, estimate_9)).findall(str(data))
                                if len(ESTIMATE) >= 1:          # Can only have estimate
                                    ESTIMATE = ESTIMATE[0]
                                else:
                                    ESTIMATE = ''
                        else:
                            ESTIMATE = ''

                        # Hammer price is € 1,483,619, € 1,483, € 134 or not sold/not listed
                        if 'Hammer' in content:
                            HAMMER_PRICE = re.compile(
                                    "(%s|%s|%s)" % (hammer_price_1, hammer_price_2, hammer_price_3)).findall(str(data))
                            if len(HAMMER_PRICE) >= 1:  # can only have hammer price
                                HAMMER_PRICE = HAMMER_PRICE[-1]

                                print([ARTIST, TITLE, DATE, HAMMER_PRICE, ESTIMATE])
                                writer.writerow([ARTIST, TITLE, DATE, HAMMER_PRICE, ESTIMATE])

                        else:
                            if len(re.compile("Not sold").findall(str(data))) >= 1:
                                NOT_HAMMER_PRICE = 'Not sold'
                            elif len(re.compile("Not listed").findall(str(data))) >= 1:
                                NOT_HAMMER_PRICE = 'Not listed'
                            else:
                                NOT_HAMMER_PRICE = ''

                            print([ARTIST, TITLE, DATE, NOT_HAMMER_PRICE, ESTIMATE])
                            writer.writerow([ARTIST, TITLE, DATE, NOT_HAMMER_PRICE, ESTIMATE])

                    sleep(6)
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
