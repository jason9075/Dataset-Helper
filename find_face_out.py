import glob

import cv2
from cv_core.detector.face_inference import FaceLocationDetector

PATH = '/Volumes/Intel_SSD/pic/麥帥一橋/10/*.JPG'
MARGIN_TOP = 200
MARGIN_LEFT = 200
MARGIN_RIGHT = 200
MARGIN_BOTTOM = 200


def main():
    detector = FaceLocationDetector()

    cnt = 0
    for file_path in glob.glob(PATH):
        filename = file_path.split('/')[-1]
        image = cv2.imread(file_path)
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
            cv2.imwrite(f'images/output/{filename[:-4]}_{i}.jpg', face_image)
        del image

        cnt += 1
        print(f'{file_path} done. ({cnt})')


if __name__ == '__main__':
    main()
