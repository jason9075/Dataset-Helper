import cv2
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("csv", help="csv file")
parser.add_argument("--data-dir", default="images/", help="img folder")
parser.add_argument("--output", default="filter.csv", help="img folder")


def main():
    arg = parser.parse_args()

    df = pd.read_csv(arg.csv)
    img_name_list = df.name.unique()
    drop_video_name = None

    print('push \'Enter\' to next, \'a\' to keep it \'z\' to drop out, \'q\' to exit! \n'
          '     \'x\' to drop all this video, \'<number>\' to jump index, \'p\' to jump previous.')

    idx = 0
    while True:
        img_name = img_name_list[idx]
        available = list(df.loc[df.name == img_name, 'available'])[0]
        if img_name[:11] == drop_video_name:
            df.loc[df.name == img_name, 'available'] = 0
            print(f'drop {img_name}')
            idx += 1
            continue
        drop_video_name = None

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

        ans = input(f'{available}:({idx + 1} / {len(img_name_list)}) {img_name}:')
        if ans.lower() == 'q':
            is_save = input('save result? (y/n)')
            if is_save.lower() == 'n':
                print('No save.')
                exit(0)
            print('Save.')
            save_result(df, arg.output)
        elif ans.lower() == 'p':
            if idx == 0:
                print('idx already zero')
                continue
            idx -= 1
            continue
        elif ans.isdigit() and 0 < int(ans) < len(img_name_list):
            idx = int(ans) - 1
            continue
        elif ans.lower() == 'x':
            drop_video_name = img_name[:11]
            df.loc[df.name == img_name, 'available'] = 0
        elif ans.lower() == 'z':
            df.loc[df.name == img_name, 'available'] = 0
        elif ans.lower() == 'a':
            df.loc[df.name == img_name, 'available'] = 1

        idx += 1
        if idx == len(img_name_list):
            break

    print('All Complete, Thanks a lot <3')
    save_result(df, arg.output)


def save_result(df, name):
    df.to_csv(name, index=0)
    exit(0)


if __name__ == '__main__':
    main()
