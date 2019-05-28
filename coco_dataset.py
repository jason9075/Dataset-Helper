import cv2
import face_recognition
import numpy as np


class PersonLocationDetector:
    def __init__(self):
        super().__init__()
        self.proto_path = 'models/MobileNetSSD_deploy.caffemodel'
        self.model_path = 'models/deploy.prototxt'
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        self.classNames = {15: 'person'}

    def predict(self, frame, ssd_thr):
        frame_resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300),
                                     (127.5, 127.5, 127.5), True)
        self.net.setInput(blob)
        # Prediction of network
        detections = self.net.forward()
        # Size of frame resize (300x300)
        rows = frame_resized.shape[0]
        cols = frame_resized.shape[1]
        output = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > ssd_thr:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label

                # Object location
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame
                heightFactor = frame.shape[0] / 300.0
                widthFactor = frame.shape[1] / 300.0
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                if class_id in self.classNames:
                    if (xLeftBottom <= 0) or (xRightTop <= 0) or (
                            yLeftBottom <= 0) or (yRightTop <= 0):  # 負座標
                        continue
                    output.append([(xLeftBottom, yLeftBottom),
                                   (xRightTop, yRightTop)])
        return output


class FaceLocationDetector:
    def predict(self, img):
        locations = face_recognition.face_locations(img)
        """
        fr 回傳格式:(top, right, bottom, left)
        故修改成 (start_x, start_y, end_x, end_y)
        """
        locations = [(loc[3], loc[0], loc[1], loc[2]) for loc in locations]
        return locations


class SsdFaceLocationDetector:
    def __init__(self):
        super().__init__()
        self.proto_path = 'models/fr_deploy.prototxt.txt'
        self.model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def predict(self, img, ssd_thr):
        face_location_list = list()
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < ssd_thr:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if (startX <= 0) or (startY <= 0) or (endX <= 0) or (endY <= 0):
                continue
            if (w < startX) or (h < startY) or (w < endX) or (h < endY):
                continue
            face_location_list.append((startX, startY, endX, endY))

        return face_location_list


def main():
    person_detector = PersonLocationDetector()
    face_detector = FaceLocationDetector()


if __name__ == '__main__':
    main()
