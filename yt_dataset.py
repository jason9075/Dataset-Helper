import glob
import os
from enum import Enum
from random import randint

from bs4 import BeautifulSoup as bs
import requests
import cv2

from pytube import YouTube

from detector import PersonLocationDetector, FaceLocationDetector

SKIP_FRAME = 100
START_ID = '8i7xNbRnvkI'
ANNO_FILE = 'annotation.csv'


class Category(Enum):
    PERSON = 1
    FACE = 2


def purge():
    for f in glob.glob('images/*.jpg'):
        os.remove(f)
    for f in glob.glob('buffer/*.mp4'):
        os.remove(f)
    if os.path.exists(ANNO_FILE):
        os.remove(ANNO_FILE)


def play_next_video(current_id):
    r = requests.get(f'https://www.youtube.com/watch?v={current_id}')
    page = r.text
    soup = bs(page, 'html.parser')
    vids = soup.findAll('a', attrs={'class': 'content-link'})
    next_id = vids[randint(0, 6)]['href']
    return next_id.split('=')[1]


def main():
    purge()

    person_detector = PersonLocationDetector()
    face_detector = FaceLocationDetector()

    video_id = START_ID
    with open(ANNO_FILE, "a") as anno_file:
        anno_file.write("name, type, start_x, start_y, end_x, end_y\n")

        for _ in range(0, 1000):
            print(f'start download video {video_id}')

            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            if 1800 < int(yt.length):
                video_id = play_next_video(video_id)
                continue
            yt.streams.first().download(output_path='buffer', filename=video_id)

            file_name = f'buffer/{video_id}.mp4'
            cap = cv2.VideoCapture(file_name)

            idx = 0
            while True:
                ret = cap.grab()
                idx += 1
                if ret is False:
                    print(f'end of video {video_id}, process to next one.')
                    break

                if idx % SKIP_FRAME != 0:
                    continue

                _, frame = cap.retrieve()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                person_loc = person_detector.predict(frame)
                if len(person_loc) == 0:  # 沒抓到人也不抓臉了
                    continue
                face_loc = face_detector.predict(frame)

                file_name = f'{video_id}_{idx}.jpg'

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                record_data(frame, anno_file, person_loc, face_loc, file_name)

                draw_info(face_loc, frame, person_loc)

                cv2.imshow('frame', frame)
                cv2.waitKey(1)

            cap.release()
            os.remove(f'buffer/{video_id}.mp4')

            video_id = play_next_video(video_id)

    cv2.destroyAllWindows()


def draw_info(face_loc, frame, person_loc):
    for person in person_loc:
        cv2.rectangle(frame, person[0], person[1],
                      (0, 255, 0), 2)
    for face in face_loc:
        cv2.rectangle(frame, face[0], face[1],
                      (255, 0, 0), 2)


def record_data(frame, anno_file, person_loc, face_loc, file_name):
    for person in person_loc:
        anno_file.write(
            f"{file_name}, {Category.PERSON.value}, {person[0][0]},{person[0][1]},{person[0][0]},{person[1][1]}\n")

    for face in face_loc:
        anno_file.write(
            f"{file_name}, {Category.FACE.value}, {face[0][0]},{face[0][1]},{face[0][0]},{face[1][1]}\n")

    cv2.imwrite(f'images/{file_name}', frame)


if __name__ == '__main__':
    main()