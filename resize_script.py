import glob

import cv2


def main():
    for img_path in glob.glob(f'../mask_checker/dataset/mask_dataset_origin/mask/*.jpg'):
        img_name = img_path.split('/')[-1]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(f'../mask_checker/dataset/office_mask/mask_dataset_origin_mask/{img_name}', image)


if __name__ == '__main__':
    main()
