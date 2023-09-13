from icrawler.builtin import GoogleImageCrawler

keyword = 'Maldives landscape'
root_dir = 'data/' + keyword

crawler = GoogleImageCrawler(
    feeder_threads = 3,
    parser_threads = 3,
    downloader_threads =3,
    storage = {'root_dir': root_dir}
)

filters = dict(
    size = 'medium'
)

crawler.crawl(keyword=keyword, filters=filters, max_num=100)