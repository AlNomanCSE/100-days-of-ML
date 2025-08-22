import pandas as pd
import requests
from bs4 import BeautifulSoup
import time


def fetchAndSaveToFile(url,path):
    r = requests.get(url)
    with open(path,"w") as f:
        f.write(r.text)

url = "https://www.thedailystar.net"

fetchAndSaveToFile(url,"data/dailyster.html")

def get_headeliens_from_file(file_path):
    try:
        with open(file_path,"r",encoding="utf-8") as file:
            html_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return[]
    soup = BeautifulSoup(html_content, "html.parser")
    headline_tags= soup.find('h1')
    headlines = [tag.get_text(strip=True) for tag in headline_tags]
    print(headlines)
    return headlines

headline_list = get_headeliens_from_file("data/dailyster.html")

if headline_list:
    for index,headline in enumerate(headline_list,1):
        print(f"{index}. {headline}")
else:
    print("Could not find any headlines. Please check the HTML tags and classes.")