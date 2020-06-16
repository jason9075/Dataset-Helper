import os

from icrawler.builtin import GoogleImageCrawler


def main():
    with open('name_list_2') as f:
        content = f.readlines()

    name_list = [x.strip() for x in content]

    for name in name_list:
        print(f'############### finding {name}')
        output_path = f'images/search/{name}'
        if os.path.isdir(output_path):
            print(f'{name} exist.')
            continue
        google_crawler = GoogleImageCrawler(feeder_threads=1,
                                            parser_threads=2,
                                            downloader_threads=4,
                                            storage={'root_dir': output_path})
        google_crawler.crawl(keyword=name, max_num=1000)
        print(f'{name} complete.')


if __name__ == '__main__':
    main()
