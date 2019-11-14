import os
from threading import Thread
from queue import Queue
import cv2
import time

q = Queue(300)
count = 0


def get_frames():
    global frame_current
    global jj
    while True:
        start = time.time()
        ret, frame_current = cap.read()
        q.put(frame_current)
        end = time.time()
        print(1/(end - start))


def show_frames():
    global frame_current
    global record_start
    while True:
        if frame_current is not None:
            cv2.imshow("video", frame_current)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                print("start recording..")
                record_start = True


def record_frames():
    global record_start
    global count
    while True:
        if record_start:
            frame = q.get()
            cv2.imwrite(os.path.join(imgs_path, "{}.jpg".format(count)), frame)
            count += 1


if __name__ == "__main__":
    width = int(1280)  # 960 540 1920 1080 1280 720
    height = int(720)
    imgs_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\web_cam_calibrate_imgs"
    frame_current = None
    record_start = False
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    th_get = Thread(target=get_frames)
    th_show = Thread(target=show_frames)
    th_record = Thread(target=record_frames)
    th_get.start()
    th_show.start()
    th_record.start()
    # th_get.join()
    # th_show.join()
    # th_record.join()
    print("hahaha...")
    # time.sleep(100)

