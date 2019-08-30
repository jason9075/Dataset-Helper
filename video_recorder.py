import cv2
from datetime import datetime
import timeit
import platform

DEVICE = 0
if platform.system() == 'Darwin':
    FOURCC = cv2.VideoWriter_fourcc(*'MP4V')
else:
    FOURCC = cv2.VideoWriter_fourcc(*'XVID')
FPS = 10
FRAME_SIZE = (1280, 720)
RECORD_DURATION = 5 * 60  # sec
WAITING_DURATION = 30 * 60  # sec


def time_to_recording(start):
    return int(timeit.default_timer() - start) % (RECORD_DURATION + WAITING_DURATION) < RECORD_DURATION


def main():
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    out = cv2.VideoWriter(f'video/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.avi', FOURCC, FPS, FRAME_SIZE)
    recording_status = True

    start_time = timeit.default_timer()

    while cap.isOpened():
        ret, frame = cap.read()

        if time_to_recording(start_time):
            if recording_status:
                pass
            else:
                recording_status = True
                out = cv2.VideoWriter(f'video/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.avi', FOURCC, FPS,
                                      FRAME_SIZE)
        else:
            if recording_status:
                recording_status = False
                out.release()
            else:
                continue

        if ret:
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
