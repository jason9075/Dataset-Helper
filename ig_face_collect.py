import glob
import json
import os
import shutil
import traceback
from multiprocessing import Pool

import cv2
import requests

from detector import FaceLocationDetector

PATH = '../instagram-crawler/output/*.json'
OUT_PATH = 'images/ig_face/'
NUM_PROCESS = 5

def main():
    detector = FaceLocationDetector()

    with Pool(NUM_PROCESS) as pool:
        for json_file_path in glob.glob(PATH):
            pool.apply_async(collect_user, args=(detector, json_file_path,), error_callback=handle_error)

        pool.close()
        pool.join()


def handle_error(e):
    traceback.print_exception(type(e), e, e.__traceback__)


def collect_user(detector, json_file_path):
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
    if len(faces) == 0:
        os.remove(head_path)

    for idx, post_url in enumerate(post_urls):
        post_path = f'{user_folder}/{idx}.jpg'
        download_image_url(post_url, post_path)
        post_image = cv2.imread(post_path)
        if post_image is None:
            continue
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)
        faces = detector.predict(post_image)
        if len(faces) == 0:
            os.remove(post_path)

    if len(glob.glob(f'{user_folder}/*.jpg')) == 0:
        print(f'{username} empty.')
        shutil.rmtree(user_folder)
        return

    print(f'{username} done.')


def download_image_url(url, path):
    img_data = requests.get(url).content
    with open(path, 'wb') as handler:
        handler.write(img_data)


if __name__ == '__main__':
    main()
