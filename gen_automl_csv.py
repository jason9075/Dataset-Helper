import glob
import os

PATH = '../store/'
PREFIX = 'gs://jasonml/store'


def main():
    labels = os.listdir(PATH)
    labels = [label for label in labels if label != '.DS_Store']
    labels = [label for label in labels if label != '1']
    labels = [label for label in labels if label != '2']
    labels = [label for label in labels if label != '3']
    labels = [label for label in labels if label != '4']
    labels = [label for label in labels if label != '5']
    labels = [label for label in labels if label != '15']
    labels = [label for label in labels if label != '16']
    labels = [label for label in labels if label != '17']
    labels = [label for label in labels if label != '18']
    labels = [label for label in labels if label != '.DS_Store']
    print(labels)

    with open('../store/labels.csv', 'a') as the_file:
        for label in labels:
            for video_path in glob.glob(os.path.join(PATH, label, '*.mp4')):
                file_name = video_path.split('/')[-1]
                the_file.write(f'{PREFIX}/{label}/{file_name},p{label},0,inf\n')


if __name__ == '__main__':
    main()
