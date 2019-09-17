import cv2
import numpy as np
import os


if __name__ == "__main__":
    left_video_name = os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\v2", "output_left.avi")
    right_video_name = os.path.join(r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\v2", "output_right.avi")
    video_left_cap = cv2.VideoCapture(left_video_name)
    video_right_cap = cv2.VideoCapture(right_video_name)
    left_save_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\left"
    right_save_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\right"
    i = 0
    while video_left_cap.isOpened() and video_right_cap.isOpened():
        ret_left, frame_left = video_left_cap.read()
        ret_right, frame_right = video_right_cap.read()
        if ret_left and ret_right:
            im_left_show = np.rot90(frame_left, 1)
            im_right_show = np.rot90(frame_right, 1)
            cv2.imwrite(os.path.join(left_save_path, "{}.jpg".format(i)), im_left_show)
            cv2.imwrite(os.path.join(right_save_path, "{}.jpg".format(i)), im_right_show)
            i += 1
            if i % 100 == 0:
                print("process {}".format(i))

