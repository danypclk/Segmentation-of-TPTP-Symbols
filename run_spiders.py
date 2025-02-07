import sys
from twisted.internet import asyncioreactor, defer

# Install the asyncio reactor before anything else is imported
try:
    asyncioreactor.install()
except Exception as e:
    print(f"Error installing asyncio reactor: {e}", file=sys.stderr)

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging

def run_all_spiders_sequentially():
    configure_logging()  # Set up logging for Scrapy
    process = CrawlerProcess(get_project_settings())  # Get Scrapy project settings

    # Get a list of all spiders in your project
    spider_loader = process.spider_loader
    spider_names = spider_loader.list()  # List all available spiders
    
    # Run spiders sequentially using deferreds
    @defer.inlineCallbacks
    def crawl_sequentially():
        for spider_name in spider_names:
            print(f"Starting spider: {spider_name}")
            yield process.crawl(spider_name)  # Wait for each spider to finish before starting the next
        process.stop()  # Stop the process after all spiders finish

    crawl_sequentially()  # Start the sequential crawling
    process.start()  # Block until all spiders are finished

if __name__ == "__main__":
    run_all_spiders_sequentially()