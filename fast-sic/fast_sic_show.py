import scipy.io as scio
import os
import numpy as np
import cv2


def load_mat(mat_file):
    data = scio.loadmat(mat_file)
    return data["current_shape"]


if __name__ == "__main__":
    mat_root = r"\\192.168.20.63\ai\Liyou_wang_data\FAST-sic-result"
    img_root = r"\\192.168.20.63\ai\face_data\20190802\image\20190802_1564716876290_ji_xian_biao_qing\resize_1_4_rot90_048170110027"
    thickness = 1
    for num in range(1, 494):
        mat_name = "{}.mat".format(num)
        jpg_name = "{}.jpg".format(num)
        mat_file = os.path.join(mat_root, mat_name)
        frame = cv2.imread(os.path.join(img_root, jpg_name))
        data = load_mat(mat_file)
        filtered_state_means0 = data - 1
        face = filtered_state_means0[:33][:, [0, 1]].astype(np.uint)
        left_brow = filtered_state_means0[33:42][:, [0, 1]].astype(np.uint)
        right_brow = filtered_state_means0[42:51][:, [0, 1]].astype(np.uint)
        nose_top = filtered_state_means0[51:55][:, [0, 1]].astype(np.uint)
        nose_bot = filtered_state_means0[55:60][:, [0, 1]].astype(np.uint)
        eye_left = filtered_state_means0[60:68][:, [0, 1]].astype(np.uint)
        eye_right = filtered_state_means0[68:76][:, [0, 1]].astype(np.uint)
        mouth_in = filtered_state_means0[88:96][:, [0, 1]].astype(np.uint)
        mouth_out = filtered_state_means0[76:88][:, [0, 1]].astype(np.uint)

        # cv2.circle(frame,  )
        cv2.polylines(frame, np.int32([face]), False, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([left_brow]), True, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([right_brow]), True, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([nose_top]), False, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([nose_bot]), False, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([eye_left]), True, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([eye_right]), True, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([mouth_out]), True, (0, 255, 0), thickness=thickness)
        cv2.polylines(frame, np.int32([mouth_in]), True, (0, 255, 0), thickness=thickness)
        cv2.imshow("check", frame)
        cv2.waitKey(0)
        print("frame id: {}".format(num))
