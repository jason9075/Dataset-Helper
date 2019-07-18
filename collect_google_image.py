import os

from icrawler.builtin import GoogleImageCrawler


def main():
    with open('name_list') as f:
        content = f.readlines()

    name_list = [x.strip() for x in content]

    for name in name_list:
        output_path = f'images/search/{name}'
        if os.path.isdir(output_path):
            print(f'{name} exist.')
            continue
        google_crawler = GoogleImageCrawler(storage={'root_dir': output_path})
        google_crawler.crawl(keyword=name, max_num=100)
        print(f'{name} complete.')


if __name__ == '__main__':
    main()
