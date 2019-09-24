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


def My_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list):
    """

    :param
    Ml: 左投影矩阵
    :param
    Mr: 右侧投影矩阵
    :param
    left_uvs_list: 左侧图像像素坐标
    :param
    right_uvs_list: 右侧图像像素坐标
    :return:
    """
    x_ = []
    for i in range(0, len(left_uvs_list)):
        # A, b = get_AB(Ml, Mr, left_uvs_list[0][0], left_uvs_list[0][1], right_uvs_list[0][0], right_uvs_list[0][1])
        A_tmp, b_tmp = get_AB(Ml, Mr, left_uvs_list[i][0], left_uvs_list[i][1], right_uvs_list[i][0],
                              right_uvs_list[i][1])
        retval, X = cv2.solve(A_tmp, b_tmp, flags=cv2.DECOMP_SVD)
        print("error is []".format(A_tmp.dot(X) - b_tmp))
        x_.append(X.ravel())
    return x_


def cv_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list):
    """
    cv2 自带的求解3D点
    :param Ml: 左投影矩阵
    :param Mr: 右侧投影矩阵
    :param left_uvs_list: 左侧图像像素坐标
    :param right_uvs_list: 右侧图像像素坐标
    :return: 计算出来的3D坐标
    """
    x_ = cv2.triangulatePoints(Ml, Mr, np.array(left_uvs_list, dtype=np.float32).T,
                               np.array(right_uvs_list, dtype=np.float32).T)
    x_ = x_.T/(x_.T[:, -1][:, np.newaxis])
    x_ = x_[:, :-1]
    return x_


def caculate_camera_Mat(pts1, pts2):
    """
    计算相机矩阵
    :param pts1:
    :param pts2:
    :return: 做鱼连个相机矩阵
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2)

    # Now decompose F to R and t using SVD
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    w, u, vt = cv2.SVDecomp(F)
    R = u * W * vt
    t = u[:, 2, np.newaxis]

    P1 = np.hstack((R, t))  # Projection matrix of second cam is ready

    P0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])  # Projection matrix of first cam at origin
    return P0, P1


if __name__ == "__main__":
    left_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\left"
    right_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\right"
    left_img = cv2.imread(os.path.join(left_img_root_path, "100.jpg"))
    right_img = cv2.imread(os.path.join(right_img_root_path, "100.jpg"))
    radius = 10
    Ml = np.array([[640, 0, 360, 0], [0, 640, 640, 0], [0, 0, 1, 0]], dtype=np.float32)
    Mr = np.array([[-23096.52209925945, 121.0767469074435, -15479.67020955832, -92.81534872139144],
                   [-6065.312586815806, -18157.9877783466, -11088.99093381785, 375.8810604672095],
                   [-17.55229880699845, -4.19159965121192, -13.41794155798649, 0.7852329199736483]], dtype=np.float32)  # 右侧相机矩阵
    left_uvs_list = [[191,193],[274,183],[355,185],[450,199],[518,206],[140,272],[503,282],[312,305],[274,527],[245,636],[284,650],[325,646],[231,729],[284,753],[349,738],[285,852]]
    right_uvs_list = [[323,218],[406,213],[482,219],[559,243],[592,254],[263,301],[597,323],[444,343],[492,574],[416,691],[459,700],[490,686],[375,781],[442,787],[490,761],[427,894]]

    x_ = np.array(My_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list), dtype=np.float32)  # my own 3d restruction code

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
    # visulize face
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
    ax.plot(x_[[5, 0, 1, 2, 3, 4, 6], 0], x_[[5, 0, 1, 2, 3, 4, 6], 1], x_[[5, 0, 1, 2, 3, 4, 6], 2], color='r')
    ax.plot(x_[[1, 7, 2], 0], x_[[1, 7, 2], 1], x_[[1, 7, 2], 2], color='r')
    ax.plot(x_[[7, 8], 0], x_[[7, 8], 1], x_[[7, 8], 2], color='r')
    ax.plot(x_[[9, 10, 11], 0], x_[[9, 10, 11], 1], x_[[9, 10, 11], 2], color='r')
    ax.plot(x_[[12, 13, 14], 0], x_[[12, 13, 14], 1], x_[[12, 13, 14], 2], color='r')
    ax.plot(x_[[13, 15], 0], x_[[13, 15], 1], x_[[13, 15], 2], color='r')

    plt.show()

