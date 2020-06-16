import argparse
import glob
import os

import cv2
import tensorflow as tf
import numpy as np

KEY_IMAGE = 'image_raw'
KEY_LABEL = 'label'
KEY_TEXT = 'text'
IMAGE_SIZE = (224, 224)


class Data:
    def __init__(self, idx, path, text):
        self.idx = idx
        self.path = path
        self.text = text

    @staticmethod
    def gen_datas(idx, paths):
        results = []
        for path in paths:
            text = path.split('/')[-1]
            results.append(Data(idx, path, text))
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--image-folder', type=str,
                        help='path to the image file')
    parser.add_argument('--output-path', default='./', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    folders = str(args.image_folder).split(',')

    cat_index = 0

    datas = []

    for folder in folders:
        faces = [
            o for o in os.listdir(folder) if os.path.isdir(os.path.join(folder, o))
        ]
        faces = {f: glob.glob(os.path.join(folder, f, '*.jpg')) for f in faces}

        for _, paths in faces.items():
            data = Data.gen_datas(cat_index, paths)
            datas = datas + data
            cat_index += 1

    np.random.shuffle(datas)

    #  write records  #
    output_path = os.path.join(args.output_path, f'train_{cat_index}.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i, data in enumerate(datas):
        img = cv2.imread(data.path)
        img = cv2.imencode('.jpg', img)[1].tostring()
        img = cv2.resize(img, IMAGE_SIZE)
        label = data.idx
        text = data.text.encode('utf8')
        example = tf.train.Example(features=tf.train.Features(feature={
            KEY_IMAGE: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            KEY_LABEL: tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            KEY_TEXT: tf.train.Feature(bytes_list=tf.train.BytesList(value=[text])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i != 0 and i % 1000 == 0:
            print('%d num of images processed.' % i)
    print(f'total {len(datas)} images process complete.')
    writer.close()


if __name__ == '__main__':
    main()
