import cv2

from detector import PersonLocationDetector, FaceLocationDetector


def main_test():
    person_detector = PersonLocationDetector()
    face_detector = FaceLocationDetector()

    image = cv2.imread('man_vs_gorilla.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    person_loc = person_detector.predict(image)
    face_loc = face_detector.predict(image)

    print(f"p:{person_loc}")
    print(f"f:{face_loc}")

    # Show image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for person in person_loc:
        cv2.rectangle(image, person[0], person[1],
                      (0, 255, 0), 2)
    for face in face_loc:
        cv2.rectangle(image, face[0], face[1],
                      (255, 0, 0), 2)
    cv2.imshow('frame', image)
    cv2.waitKey(1)

    input()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_test()
