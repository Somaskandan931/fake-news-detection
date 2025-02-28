import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from scrapy.selector import Selector

class AltnewsSpider(scrapy.Spider):
    name = "altnews"
    start_urls = ["https://altnews.in"]

    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in the background
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def parse(self, response):
        self.driver.get(response.url)
        html = self.driver.page_source
        sel = Selector(text=html)

        # Extract data (modify according to what you need)
        headlines = sel.css("h2 a::text").getall()
        for headline in headlines:
            yield {"headline": headline}

    def closed(self, reason):
        self.driver.quit()
