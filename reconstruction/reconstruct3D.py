import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


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

    ul = 40  # 左侧图片像素坐标
    vl = 60
    ur = 30  # 右侧图片像素坐标
    vr = 20
    Ml = np.array([[10, 0, 360, 0], [0, 10, 640, 0], [0, 0, 1, 0]], dtype=np.float32)
    Mr = np.array([[5.241245912283468, -4.804079641277808, -15.62010463754545, 20.71041494220587],
                   [5.012054817985312, -7.51852747759903, -47.34634706530864, 42.5533817579683],
                   [0.008611215279209808, -0.01194003359547416, -0.07051263005552549, 0.05128554702553706]], dtype=np.float32)  # 右侧相机矩阵
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
        print("left:{} {}".format(i, uvs))
        uvs = Mr.dot(xw[:, np.newaxis])
        uvs = uvs / uvs[2]
        print("right:{} {}".format(i, uvs))
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
    plt.show()

