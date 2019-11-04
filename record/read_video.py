import cv2
import time
import numpy as np


if __name__ == "__main__":
    width = int(1280)  # 960 540 1920 1080 1280 720
    height = int(720)
    cap_left = cv2.VideoCapture(0)  # 调整左右
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_left = cv2.VideoWriter('./output/output_left.avi', fourcc, 15.0, (width, height))
    recording = False
    while True:
        # get a frame
        start = time.time()
        ret_left, frame_left = cap_left.read()

        cv2.imshow("capture_left", frame_left)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            recording = True
            print("start recording")
        elif key & 0xFF == ord('q'):
            print("break")
            break
        if recording:
            out_left.write(frame_left)
            end = time.time()
            print("time is {}".format(end - start))

    cap_left.release()
    cv2.destroyAllWindows()