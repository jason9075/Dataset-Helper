import glob
import os
import uuid
from shutil import copyfile

OUTPUT = 'output/'


def main():
    files = glob.glob('/mnt/asia_sr_face/*')
    for file in files:
        images = glob.glob(os.path.join(file, '*.jpg'))
        for image in images:
            copyfile(image, f'OUTPUT/{uuid.uuid4()}.jpg')


if __name__ == '__main__':
    main()
