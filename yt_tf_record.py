import hashlib
import os
from argparse import ArgumentParser

import cv2
import pandas as pd
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument(
    "--csv", default="annotation.csv", help="annotation file")
parser.add_argument(
    "--data-dir", default="images/",
    help="image folder")
parser.add_argument(
    "--output-name", default="annotation", help="annotation file")
parser.add_argument(
    "--output-path", default="output_path/", help="output_path")
parser.add_argument(
    "--label-map", default="label_map.pbtxt", help="Path to label map proto")


def df_to_tf_example(img_name, img_size, encoded_image, obj_df):
    key = hashlib.sha256(encoded_image).hexdigest()

    height, width = img_size

    start_x = []
    start_y = []
    end_x = []
    end_y = []
    classes = []
    classes_text = []

    for _, obj in obj_df.iterrows():
        start_x.append(float(obj['start_x']) / width)
        start_y.append(float(obj['start_y']) / height)
        end_x.append(float(obj['end_x']) / width)
        end_y.append(float(obj['end_y']) / height)
        label = 'person' if obj['type'] == 1 else 'face'
        classes_text.append(label.encode('utf8'))
        classes.append(int(obj['type']))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(
            img_name.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_image),
        'image/object/bbox/xmin': float_list_feature(start_x),
        'image/object/bbox/xmax': float_list_feature(end_x),
        'image/object/bbox/ymin': float_list_feature(start_y),
        'image/object/bbox/ymax': float_list_feature(end_y),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return example


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df['available'] == 1]

    output_path = os.path.join(args.output_path, f'{args.output_name}.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)
    print(f'write to {output_path}.')

    img_name_list = df.name.unique()
    print(f"{len(img_name_list)} of images and {len(df.index)} of obj have to record.")
    for img_name in img_name_list:
        obj_df = df.loc[(df.name == img_name)]

        with tf.gfile.GFile(f'{args.data_dir}{img_name}', 'rb') as fid:
            encoded_image = fid.read()
        img = cv2.imread(f'{args.data_dir}{img_name}')
        size = img.shape[0:2]
        tf_example = df_to_tf_example(img_name, size, encoded_image, obj_df)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"complete file: {output_path}")


if __name__ == '__main__':
    main()
