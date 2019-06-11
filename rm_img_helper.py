import os
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("csv", help="csv file")
parser.add_argument("--data-dir", default="images/", help="img folder")


def main():
    arg = parser.parse_args()

    df = pd.read_csv(arg.csv)
    df = df[df['available'] == 0]
    img_name_list = df.name.unique()

    for idx, img_name in enumerate(img_name_list):
        print(f'rm: {arg.data_dir}{img_name}')
        os.remove(f'{arg.data_dir}{img_name}')


if __name__ == '__main__':
    main()
