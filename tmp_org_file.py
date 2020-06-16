import glob
import shutil

def main():
    for path in glob.glob('images/search/google/**/*.jpg'):
        filename = path.split('/')[-1]
        category = path.split('/')[-2]
        shutil.copy2(path, f'images/search/all/{category}_{filename}')


if __name__ == '__main__':
    main()
