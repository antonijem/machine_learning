import scrapy
from urllib.parse import urljoin
import re
from ..items import PszprojekatItem

class PszSpiderSpider(scrapy.Spider):
    name = "psz_spider"
    allowed_domains = ["knjizare-vulkan.rs"]
    start_urls = [
        "https://www.knjizare-vulkan.rs/domace-knjige",
        "https://www.knjizare-vulkan.rs/strana-izdanja-knjige"
    ]
    page_limits = {
        "https://www.knjizare-vulkan.rs/domace-knjige/": 1039, #1037
        "https://www.knjizare-vulkan.rs/strana-izdanja-knjige/": 137 # 132
    }

    def parse(self, response):
        base_url = response.url.split('/page-')[0] + '/'
        page_number = response.meta.get('page_number', 0)
        print(f"{base_url}")
        page_limit = self.page_limits.get(base_url, 1)


        book_links = response.css(".product-link::attr(href)").getall()
        print(book_links)
        for book_link in book_links:
            yield response.follow(book_link, callback=self.parse_book)

        print(f"Current page: {page_number} of {page_limit}")  # Debug print statement

        if page_number < page_limit:
            next_page = urljoin(base_url, f'page-{page_number + 1}')
            print(f"Trying to follow: {next_page}")  # Debug print statement
            yield response.follow(next_page, callback=self.parse, meta={'page_number': page_number + 1})
    def parse_book(self, response):
        items = PszprojekatItem()
        raw_description = response.css("#tab_product_description").get()
        cleaned_description = self.clean_html(raw_description)

        author = response.css(".product-details-info div .atributs-wrapper a::text").get()
        publisher = response.css(".chosen-atributes a::text").get()
        if author is None or publisher is None:
            pages = response.css("tr:nth-child(8) td+ td::text").get()
        else:
            pages = response.css("tr:nth-child(9) td+ td::text").get()

        items['title'] = response.css('.product-details-info .title span::text').get()
        items['price'] = response.css('.product-price-value::text').getall()[1].strip()
        items['author'] = author
        items['category'] = response.css(".category a::text").get().strip()
        items['publisher'] = publisher
        items['year'] = response.css(".attr-povez+ tr td+ td::text").get()
        items['pages'] = pages
        items['cover'] = response.css(".attr-povez td+ td::text").get().strip()
        items['format'] = response.css("tr:nth-child(8) td+ td::text").get()
        items['description'] = cleaned_description

        yield items

    def clean_html(self, raw_html):
        clean_text = re.sub(r'<!--.*?-->', '', raw_html)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = clean_text.replace('\r\n', '').replace('\n', '').strip()
        return clean_text
