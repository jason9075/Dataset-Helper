import cv2
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("csv", help="csv file")
parser.add_argument("--data-dir", default="images/", help="img folder")


def main():
    arg = parser.parse_args()

    df = pd.read_csv(arg.csv)
    df = df[df['available'] == 1]
    img_name_list = df.name.unique()

    print('push \'Enter\' to next, \'z\' to drop out, or \'q\' to exit! ')
    for idx, img_name in enumerate(img_name_list):
        person_df = df.loc[(df.name == img_name) & (df['type'] == 1)]
        face_df = df[(df.name == img_name) & (df['type'] == 2)]

        img = cv2.imread(f'{arg.data_dir}{img_name}')

        for _, person in person_df.iterrows():
            cv2.rectangle(img, (person['start_x'], person['start_y']),
                          (person['end_x'], person['end_y']), (0, 255, 0), 2)
        for _, face in face_df.iterrows():
            cv2.rectangle(img, (face['start_x'], face['start_y']),
                          (face['end_x'], face['end_y']), (255, 0, 0), 2)

        cv2.imshow('frame', img)
        cv2.waitKey(1)

        ans = input(f'{idx+1}/{len(img_name_list)}')
        if ans.lower() == 'q':
            is_save = input('save result? (y/n)')
            if is_save.lower() == 'n':
                print('No save.')
                exit(0)
            print('Save.')
            save_result(df)
        elif ans.lower() == 'z':
            df.loc[df.name == img_name, 'available'] = 0

    print('All Complete, Thanks a lot <3')
    save_result(df)


def save_result(df):
    df.to_csv('filter.csv', index=0)
    exit(0)


if __name__ == '__main__':
    main()
