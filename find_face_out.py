import glob
import os

import cv2
from cv_core.detector.face_inference import FaceLocationDetector

PATH = './images/search/'
OUT_PATH = './images/search_face/'
MARGIN_TOP = 100
MARGIN_LEFT = 100
MARGIN_RIGHT = 100
MARGIN_BOTTOM = 100


def main():
    detector = FaceLocationDetector()

    cnt = 0
    for star_name in os.listdir(PATH):
        star_folder = f'{OUT_PATH}{star_name}'
        if os.path.isdir(star_folder) or star_name.startswith('.'):
            print(f'{star_folder} exist.')
            continue
        else:
            os.mkdir(star_folder)

        for img_path in glob.glob(f'{PATH}{star_name}/*.jpg'):
            img_name = img_path.split('/')[-1]

            image = cv2.imread(img_path)
            if image is None or image.shape[2] != 3:  # jpg have issue
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:-1]

            faces = detector.predict(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for i, face_loc in enumerate(faces):
                (start_x, start_y, end_x, end_y) = face_loc
                (start_x, start_y, end_x, end_y) = (
                    max(0, start_x - MARGIN_LEFT), max(0, start_y - MARGIN_TOP), min(w, end_x + MARGIN_RIGHT),
                    min(h, end_y + MARGIN_BOTTOM))
                face_image = image[start_y:end_y, start_x:end_x, :]
                cv2.imwrite(f'{OUT_PATH}/{star_name}/{img_name[:-4]}_{i}.jpg', face_image)

            print(f'{star_name} finish.')
            del image

        cnt += 1
        print(f'{star_name} done. ({cnt})')


if __name__ == '__main__':
    main()
