import glob
import json
import os
import shutil
import traceback
import uuid
from multiprocessing import Pool

import cv2
import numpy as np
import requests

from detector import FaceLocationDetector, FaceVectorEncoder

MARGIN_TOP = 50
MARGIN_LEFT = 50
MARGIN_RIGHT = 50
MARGIN_BOTTOM = 50

PATH = '../instagram-crawler/output/*.json'
OUT_PATH = 'images/ig_face/'
NUM_PROCESS = 5


def main():
    detector = FaceLocationDetector()
    encoder = FaceVectorEncoder()

    with Pool(NUM_PROCESS) as pool:
        for json_file_path in glob.glob(PATH):
            pool.apply_async(collect_user, args=(detector, encoder, json_file_path,), error_callback=handle_error)

        pool.close()
        pool.join()


def handle_error(e):
    traceback.print_exception(type(e), e, e.__traceback__)


def collect_user(detector, encoder, json_file_path):
    face_history = {}
    username = (json_file_path.split('/')[-1])[:-5]
    user_folder = f'{OUT_PATH}{username}'
    if os.path.isdir(user_folder):
        print(f'{username} exist.')
        return

    post_urls = []
    with open(json_file_path) as json_file:
        data = json.load(json_file)
        if not data:
            return

        head_url = data['photo_url']

        for p in data['posts']:
            post_urls.append(p['img_url'])

    os.mkdir(user_folder)
    print(f'download user: {username}.')

    head_path = f'{user_folder}/head.jpg'
    download_image_url(head_url, head_path)
    head_image = cv2.imread(head_path)
    head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
    faces = detector.predict(head_image)
    separate2faces(head_image, faces, user_folder, '0_head', encoder, face_history)
    os.remove(head_path)

    for idx, post_url in enumerate(post_urls):
        idx += 1
        post_path = f'{user_folder}/{idx}.jpg'
        download_image_url(post_url, post_path)
        post_image = cv2.imread(post_path)
        if post_image is None:
            continue
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)
        faces = detector.predict(post_image)
        separate2faces(post_image, faces, user_folder, f'{idx}_post', encoder, face_history)
        os.remove(post_path)

    if len(glob.glob(f'{user_folder}/*.jpg')) == 0:
        print(f'{username} empty.')
        shutil.rmtree(user_folder)
        return

    print(f'{username} done.')


def separate2faces(image, faces, output_folder, prefix_name, encoder, history):
    h, w = image.shape[:-1]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, face_loc in enumerate(faces):
        (start_x, start_y), (end_x, end_y) = face_loc
        face_image = image[start_y:end_y, start_x:end_x, :]
        vector = encoder.predict(face_image)
        similar_id = get_uuid(history, vector)
        (start_x, start_y, end_x, end_y) = (
            max(0, start_x - MARGIN_LEFT), max(0, start_y - MARGIN_TOP), min(w, end_x + MARGIN_RIGHT),
            min(h, end_y + MARGIN_BOTTOM))
        face_image = image[start_y:end_y, start_x:end_x, :]
        cv2.imwrite(f'{output_folder}/{similar_id}_{prefix_name}_{i}.jpg', face_image)


def euclidean_distance(feature_1, feature_2):
    return np.linalg.norm(feature_1 - feature_2)


def get_uuid(history, vector):
    dist_dict = {k: euclidean_distance(vector, v) for (k, v) in history.items()}
    if len(dist_dict) == 0:
        new_id = str(uuid.uuid4())[0:6]
        history[new_id] = vector
        return new_id
    min_key = min(dist_dict, key=dist_dict.get)

    if dist_dict[min_key] < 0.5:
        return min_key
    else:
        new_id = str(uuid.uuid4())[0:6]
        history[new_id] = vector
        return new_id


def download_image_url(url, path):
    img_data = requests.get(url).content
    with open(path, 'wb') as handler:
        handler.write(img_data)


if __name__ == '__main__':
    main()
