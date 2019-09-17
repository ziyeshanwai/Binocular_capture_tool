import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import os


def get_AB(Ml, Mr, ul, vl, ur, vr):
    """

    :param Ml: 左侧相机矩阵
    :param Mr: 右侧相机矩阵
    :param ul: 左侧相机图片的对应点坐标x
    :param vl: 左侧相机图片的对应点坐标y
    :param ur: 右侧相机图片的对应点坐标x
    :param vr: 右侧相机图片的对应点坐标y
    :return: A,B (4, 3) (4, 1)
    """
    A = np.array([[Ml[0, 0] - Ml[2, 0] * ul, Ml[0, 1] - Ml[2, 1] * ul, Ml[0, 2] - Ml[2, 2] * ul],
                  [Ml[1, 0] - Ml[2, 0] * vl, Ml[1, 1] - Ml[2, 1] * vl, Ml[1, 2] - Ml[2, 2] * vl],
                  [Mr[0, 0] - Mr[2, 0] * ur, Mr[0, 1] - Mr[2, 1] * ur, Mr[0, 2] - Mr[2, 2] * ur],
                  [Mr[1, 0] - Mr[2, 0] * vr, Mr[1, 1] - Mr[2, 1] * vr, Mr[1, 2] - Mr[2, 2] * vr],
                  ], dtype=np.float32)

    B = np.array([[Ml[2, 3] * ul - Ml[0, 3]], [Ml[2, 3] * vl - Ml[1, 3]],
                  [Mr[2, 3] * ur - Mr[0, 3]], [Mr[2, 3] * vr - Mr[1, 3]]], dtype=np.float32)
    return A, B


if __name__ == "__main__":
    left_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\left"
    right_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\right"
    left_img = cv2.imread(os.path.join(left_img_root_path, "0.jpg"))
    right_img = cv2.imread(os.path.join(right_img_root_path, "0.jpg"))
    radius = 10
    Ml = np.array([[10, 0, 360, 0], [0, 10, 640, 0], [0, 0, 1, 0]], dtype=np.float32)
    Mr = np.array([[0.3520066989851156, 2.073537362795778, -18.45796152964473, -28.37748590946849],
                   [2.508549580148226, 2.247961998832718, -13.69927652036192, -43.29439392369385],
                   [0.00274310953611928, 0.00441668809134959, -0.03340103384464285, -0.0576053978033833]], dtype=np.float32)  # 右侧相机矩阵
    left_uvs_list = [[185, 186], [271, 199], [366, 201], [467, 191], [536, 201], [152, 265], [508, 280], [316, 321], [279, 540], [255, 665], [287,678],[320,669],[239,723],[291,747],[350,742],[289,846]]
    right_uvs_list = [[298, 213], [397, 235], [485, 243], [557, 241], [594, 251], [249, 296], [586, 324], [442, 361], [491, 583], [420, 716], [457,722],[481,707],[386,785],[445,791],[489,774],[430,894]]
    x_ = []
    for i in range(0, len(left_uvs_list)):
        # A, b = get_AB(Ml, Mr, left_uvs_list[0][0], left_uvs_list[0][1], right_uvs_list[0][0], right_uvs_list[0][1])
        A_tmp, b_tmp = get_AB(Ml, Mr, left_uvs_list[i][0], left_uvs_list[i][1], right_uvs_list[i][0], right_uvs_list[i][1])
        retval, X = cv2.solve(A_tmp, b_tmp, flags=cv2.DECOMP_SVD)
        print("error is {}".format(A_tmp.dot(X)-b_tmp))
        x_.append(X.ravel())
        # A = np.vstack((A, A_tmp))
        # b = np.vstack((b, b_tmp))
    # print("A shape is {}".format(A.shape))
    # print("b shape is {}".format(b.shape))

    print("retval is {}".format(retval))
    x_ = np.array(x_, dtype=np.float32)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, len(x_)):
        xw = np.append(x_[i], 1)
        uvs = Ml.dot(xw[:, np.newaxis])
        uvs = uvs/uvs[2]
        cv2.circle(left_img, (int(uvs[0]), int(uvs[1])), radius, (255, 0, 0), 1)
        cv2.putText(left_img, "{}".format(i), (int(uvs[0] - radius / 2), int(uvs[1] + radius / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), lineType=cv2.LINE_AA)
        print("left:{} {}".format(i, uvs))
        uvs = Mr.dot(xw[:, np.newaxis])
        uvs = uvs / uvs[2]
        cv2.circle(right_img, (int(uvs[0]), int(uvs[1])), radius, (255, 0, 0), 1)
        cv2.putText(right_img, "{}".format(i), (int(uvs[0] - radius / 2), int(uvs[1] + radius / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), lineType=cv2.LINE_AA)
        print("right:{} {}".format(i, uvs))
    cv2.imshow('left_image', left_img)
    cv2.imshow('right_image', right_img)
    # cv2.waitKey(0)
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
    plt.show()

